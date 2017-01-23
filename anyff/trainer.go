package anyff

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
)

// A Batch stores an input and output batch in a packed
// format.
type Batch struct {
	Inputs  *anydiff.Const
	Outputs *anydiff.Const
	Num     int
}

// A Trainer can construct batches, compute gradients, and
// tally up costs for feed-forward neural networks.
type Trainer struct {
	Net    anynet.Layer
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
		panic("empty batch")
	}

	l := s.(SampleList)
	var ins, outs []anyvec.Vector
	for i := 0; i < l.Len(); i++ {
		sample := l.GetSample(i)
		ins = append(ins, sample.Input)
		outs = append(outs, sample.Output)
	}

	joinedIns := ins[0].Creator().Concat(ins...)
	joinedOuts := outs[0].Creator().Concat(outs...)

	return &Batch{
		Inputs:  anydiff.NewConst(joinedIns),
		Outputs: anydiff.NewConst(joinedOuts),
		Num:     l.Len(),
	}, nil
}

// TotalCost computes the total cost for the batch.
func (t *Trainer) TotalCost(b *Batch) anydiff.Res {
	outRes := t.Net.Apply(b.Inputs, b.Num)
	cost := t.Cost.Cost(b.Outputs, outRes, b.Num)
	total := anydiff.Sum(cost)
	if t.Average {
		divisor := 1 / float64(cost.Output().Len())
		return anydiff.Scale(total, total.Output().Creator().MakeNumeric(divisor))
	} else {
		return total
	}
}

// Gradient computes the gradient for the batch's cost.
// It also assigns g.LastCost to the numerical value of
// the total cost.
//
// The b argument must be a *Batch.
func (t *Trainer) Gradient(b anysgd.Batch) anydiff.Grad {
	res := anydiff.Grad{}
	for _, p := range t.Params {
		res[p] = p.Vector.Creator().MakeVector(p.Vector.Len())
	}

	cost := t.TotalCost(b.(*Batch))
	t.LastCost = anyvec.Sum(cost.Output())

	// Scale the upstream vector so that it's as if we took
	// the average of the cost.
	c := cost.Output().Creator()
	data := c.MakeNumericList([]float64{1})
	upstream := cost.Output().Creator().MakeVectorData(data)
	cost.Propagate(upstream, res)

	return res
}
