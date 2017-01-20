package anyff

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
)

// A Gradienter computes gradients for a feed-forward
// neural network.
type Gradienter struct {
	Net    anynet.Layer
	Cost   anynet.Cost
	Params []*anydiff.Var
}

// Gradient computes the gradient for the average cost of
// the batch.
//
// The anysgd.SampleList must be a SampleList.
func (g *Gradienter) Gradient(s anysgd.SampleList) anydiff.Grad {
	res := anydiff.Grad{}
	for _, p := range g.Params {
		res[p] = p.Vector.Creator().MakeVector(p.Vector.Len())
	}

	l := s.(SampleList)
	var ins, outs []anyvec.Vector
	for i := 0; i < l.Len(); i++ {
		sample := l.GetSample(i)
		ins = append(ins, sample.Input)
		outs = append(outs, sample.Output)
	}

	if len(ins) == 0 {
		return res
	}

	joinedIns := ins[0].Creator().Concat(ins...)
	inRes := anydiff.NewConst(joinedIns)
	outRes := g.Net.Apply(inRes, l.Len())

	joinedOuts := outs[0].Creator().Concat(outs...)
	desiredRes := anydiff.NewConst(joinedOuts)

	cost := g.Cost.Cost(desiredRes, outRes, l.Len())
	sum := anydiff.SumCols(&anydiff.Matrix{
		Data: cost,
		Rows: 1,
		Cols: cost.Output().Len(),
	})

	// Scale the upstream vector so that it's as if we took
	// the average of the cost.
	upstream := sum.Output().Creator().MakeVector(1)
	upstream.AddScaler(upstream.Creator().MakeNumeric(1 / float64(l.Len())))

	cost.Propagate(upstream, res)
	return res
}
