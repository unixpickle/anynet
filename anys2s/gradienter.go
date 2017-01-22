package anys2s

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
)

// A Gradienter computes gradients for a recurrent neural
// network.
type Gradienter struct {
	Func   func(anyseq.Seq) anyseq.Seq
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

	input, desired := inOutSeqs(s.(SampleList))
	actual := g.Func(input)

	if len(actual.Output()) != len(desired.Output()) {
		panic("mismatching actual and desired sequence shapes")
	}

	var idx int
	var totalCosts int
	var costSum float64
	allCosts := anyseq.Map(actual, func(a anydiff.Res, n int) anydiff.Res {
		batch := desired.Output()[idx]
		if batch.NumPresent() != n {
			panic("mismatching actual and desired sequence shapes")
		}
		totalCosts += n
		idx++
		c := g.Cost.Cost(anydiff.NewConst(batch.Packed), a, n)
		costSum += floatSum(c.Output())
		return c
	})

	g.LastCost = costSum / float64(totalCosts)

	// Configure the upstream sequence so that we are
	// effectively taking an average.
	upstream := make([]*anyseq.Batch, len(allCosts.Output()))
	var timestepUpstream anyvec.Vector
	for i, x := range allCosts.Output() {
		if timestepUpstream == nil {
			timestepUpstream = x.Packed.Creator().MakeVector(x.Packed.Len())
			avgScale := timestepUpstream.Creator().MakeNumeric(1 / float64(totalCosts))
			timestepUpstream.AddScaler(avgScale)
		} else if timestepUpstream.Len() != x.Packed.Len() {
			timestepUpstream = timestepUpstream.Slice(0, x.Packed.Len())
		}
		upstream[i] = &anyseq.Batch{
			Packed:  timestepUpstream,
			Present: x.Present,
		}
	}

	allCosts.Propagate(upstream, res)

	return res
}

func inOutSeqs(s SampleList) (in, out anyseq.Seq) {
	ins := make([][]anyvec.Vector, s.Len())
	outs := make([][]anyvec.Vector, s.Len())
	for i := 0; i < s.Len(); i++ {
		sample := s.GetSample(i)
		ins[i] = sample.Input
		outs[i] = sample.Output
	}
	return anyseq.ConstSeqList(ins), anyseq.ConstSeqList(outs)
}

func floatSum(cost anyvec.Vector) float64 {
	sum := anyvec.Sum(cost)
	switch sum := sum.(type) {
	case float32:
		return float64(sum)
	case float64:
		return sum
	default:
		return 0
	}
}
