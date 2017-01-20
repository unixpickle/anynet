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

	// After every gradient computation, LastCost is set to
	// the average cost from the batch.
	// However, if the numeric type is not float32 or
	// float64, then it is never set.
	LastCost float64
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

	// Scale the upstream vector so that it's as if we took
	// the average of the cost.
	upstream := cost.Output().Creator().MakeVector(cost.Output().Len())
	upstream.AddScaler(upstream.Creator().MakeNumeric(1 / float64(l.Len())))

	cost.Propagate(upstream, res)
	g.LastCost = floatSum(cost.Output())

	return res
}

func floatSum(cost anyvec.Vector) float64 {
	switch data := cost.Data().(type) {
	case []float32:
		var sum float32
		for _, x := range data {
			sum += x
		}
		return float64(sum)
	case []float64:
		var sum float64
		for _, x := range data {
			sum += x
		}
		return sum
	default:
		return 0
	}
}
