package anyctc

import (
	"errors"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
)

// A Batch stores a batch of input sequences and the
// corresponding labels for each.
type Batch struct {
	Inputs anyseq.Seq
	Labels [][]int
}

// A Trainer creates batches, computes gradients, and adds
// up costs for CTC.
type Trainer struct {
	Func   func(anyseq.Seq) anyseq.Seq
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
	outs := make([][]int, l.Len())
	for i := 0; i < l.Len(); i++ {
		sample, err := l.GetSample(i)
		if err != nil {
			return nil, essentials.AddCtx("fetch batch", err)
		}
		ins[i] = sample.Input
		outs[i] = sample.Label
	}
	return &Batch{
		Inputs: anyseq.ConstSeqList(l.Creator(), ins),
		Labels: outs,
	}, nil
}

// TotalCost computes the total cost for the batch.
//
// For more information on how this works, see Cost().
func (t *Trainer) TotalCost(b *Batch) anydiff.Res {
	actual := t.Func(b.Inputs)
	costs := Cost(actual, b.Labels)
	sum := anydiff.Sum(costs)
	if t.Average {
		scaler := sum.Output().Creator().MakeNumeric(1 / float64(costs.Output().Len()))
		return anydiff.Scale(sum, scaler)
	} else {
		return sum
	}
}

// Gradient computes the gradient for the batch's cost.
// It also sets t.LastCost to the numerical value of the
// total cost.
//
// The b argument must be a *Batch.
func (t *Trainer) Gradient(b anysgd.Batch) anydiff.Grad {
	res := anydiff.NewGrad(t.Params...)

	cost := t.TotalCost(b.(*Batch))
	t.LastCost = anyvec.Sum(cost.Output())

	c := cost.Output().Creator()
	data := c.MakeNumericList([]float64{1})
	upstream := c.MakeVectorData(data)
	cost.Propagate(upstream, res)

	return res
}
