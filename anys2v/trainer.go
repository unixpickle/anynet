package anys2v

import (
	"errors"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
)

// A Batch stores an input and output batch in a packed
// format.
// No input sequences may be empty.
type Batch struct {
	Inputs  anyseq.Seq
	Outputs *anydiff.Const
}

// A Trainer creates batches, computes gradients, and adds
// up costs for a sequence-to-sequence mapping.
type Trainer struct {
	Func   func(anyseq.Seq) anydiff.Res
	Cost   anynet.Cost
	Params []*anydiff.Var

	// Average indicates whether or not the total cost should
	// be averaged before computing gradients.
	// This affects gradients, LastCost, and the output of
	// TotalCost().
	Average bool

	// After every gradient computation, LastCost is set to
	// the cost from the batch.
	LastCost anyvec.Numeric
}

// Fetch produces a *Batch for the subset of samples.
// The s argument must implement SampleList.
// The batch may not be empty.
func (t *Trainer) Fetch(s anysgd.SampleList) (anysgd.Batch, error) {
	if s.Len() == 0 {
		return nil, errors.New("fetch batch: empty batch")
	}
	l := s.(SampleList)
	ins := make([][]anyvec.Vector, l.Len())
	outs := make([]anyvec.Vector, l.Len())
	for i := 0; i < l.Len(); i++ {
		sample, err := l.GetSample(i)
		if err != nil {
			return nil, essentials.AddCtx("fetch batch", err)
		}
		if len(sample.Input) == 0 {
			return nil, errors.New("fetch batch: empty sequence")
		}
		ins[i] = sample.Input
		outs[i] = sample.Output
	}
	cr := outs[0].Creator()
	return &Batch{
		Inputs:  anyseq.ConstSeqList(cr, ins),
		Outputs: anydiff.NewConst(cr.Concat(outs...)),
	}, nil
}

// TotalCost computes the total cost for the *Batch.
func (t *Trainer) TotalCost(batch anysgd.Batch) anydiff.Res {
	b := batch.(*Batch)
	n := 0
	if len(b.Inputs.Output()) > 0 {
		n = b.Inputs.Output()[0].NumPresent()
	}
	outRes := t.Func(b.Inputs)
	cost := t.Cost.Cost(b.Outputs, outRes, n)
	total := anydiff.Sum(cost)
	if t.Average {
		divisor := 1 / float64(cost.Output().Len())
		return anydiff.Scale(total, total.Output().Creator().MakeNumeric(divisor))
	} else {
		return total
	}
}

// Gradient computes the gradient for the batch's cost.
// It also sets t.LastCost to the numerical value of the
// total cost.
//
// The b argument must be a *Batch.
func (t *Trainer) Gradient(b anysgd.Batch) anydiff.Grad {
	grad, lc := anysgd.CosterGrad(t, b, t.Params)
	t.LastCost = lc
	return grad
}
