package anysgd

import (
	"math"
	"math/rand"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
)

// Shuffle shuffles a list of samples.
// If the list implements PostShuffler, then PostShuffle
// is called after the shuffle completes.
func Shuffle(s SampleList) {
	for i := 0; i < s.Len(); i++ {
		j := i + rand.Intn(s.Len()-i)
		s.Swap(i, j)
	}
	if p, ok := s.(PostShuffler); ok {
		p.PostShuffle()
	}
}

// LengthSampleList is a SampleList that carries no
// information other than its length.
//
// This may be used in conjunction with a Fetcher that
// generates data dynamically.
type LengthSampleList int

// Len returns int(l).
func (l LengthSampleList) Len() int {
	return int(l)
}

// Swap does nothing.
func (l LengthSampleList) Swap(i, j int) {
}

// Slice creates a LengthSampleList with the new size.
func (l LengthSampleList) Slice(i, j int) SampleList {
	return LengthSampleList(j - i)
}

// A ConstRater is a Rater which always returns the same
// constant learning rate.
type ConstRater float64

// Rate returns float64(c).
func (c ConstRater) Rate(epoch float64) float64 {
	return float64(c)
}

// An ExpRater is a Rater which returns
//
//     Bias + Coeff*Decay^t
//
// This is a standard kind of exponential decay schedule
// used for SGD.
type ExpRater struct {
	Bias  float64
	Coeff float64
	Decay float64
}

// Rate computes the rate for time t.
func (e *ExpRater) Rate(t float64) float64 {
	return e.Bias + e.Coeff*math.Pow(e.Decay, t)
}

// CosterGrad computes a gradient and a cost for the
// batch.
func CosterGrad(c Coster, b Batch, params []*anydiff.Var) (anydiff.Grad,
	anyvec.Numeric) {
	grad := anydiff.NewGrad(params...)
	cost := c.TotalCost(b.(*Batch))
	cr := cost.Output().Creator()
	data := cr.MakeNumericList([]float64{1})
	upstream := cr.MakeVectorData(data)
	cost.Propagate(upstream, grad)
	return grad, anyvec.Sum(cost.Output())
}

func copyGrad(g anydiff.Grad) anydiff.Grad {
	res := anydiff.Grad{}
	for va, vec := range g {
		res[va] = vec.Copy()
	}
	return res
}

func scaleGrad(g anydiff.Grad, s float64) {
	for _, v := range g {
		g.Scale(v.Creator().MakeNumeric(s))
		return
	}
}

func valueOrDefault(val, def float64) float64 {
	if val != 0 {
		return val
	} else {
		return def
	}
}
