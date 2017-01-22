package anysgd

import (
	"math"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
)

const (
	adamDefaultDecayRate1 = 0.9
	adamDefaultDecayRate2 = 0.999
	adamDefaultDamping    = 1e-8
)

// Adam implements the adaptive moments SGD technique
// described in https://arxiv.org/pdf/1412.6980.pdf.
//
// Most of this code is taken from
// https://github.com/unixpickle/sgd/blob/0e3d4c9d317b1095d02febdaedf802f6d1dbd5b1/adam.go.
type Adam struct {
	// These are decay rates for the first and second
	// moments of the gradient.
	// If these are 0, defaults as suggested in the
	// original Adam paper are used.
	DecayRate1, DecayRate2 float64

	// Damping is used to prevent divisions by zero.
	// This should be very small.
	// If it is 0, a default is used.
	Damping float64

	firstMoment  anydiff.Grad
	secondMoment anydiff.Grad
	iteration    float64
}

func (a *Adam) Transform(realGrad anydiff.Grad) anydiff.Grad {
	a.updateMoments(realGrad)

	a.iteration++
	scalingFactor := math.Sqrt(1-math.Pow(a.decayRate(2), a.iteration)) /
		(1 - math.Pow(a.decayRate(1), a.iteration))
	damping := a.damping()
	for variable, vec := range realGrad {
		firstVec := a.firstMoment[variable]
		secondVec := a.secondMoment[variable]

		vec.Set(firstVec)
		vec.Scale(vec.Creator().MakeNumeric(scalingFactor))

		divisor := secondVec.Copy()
		divisor.AddScaler(divisor.Creator().MakeNumeric(damping))
		anyvec.Pow(divisor, divisor.Creator().MakeNumeric(0.5))
		vec.Div(divisor)
	}

	return realGrad
}

func (a *Adam) updateMoments(grad anydiff.Grad) {
	if a.firstMoment == nil {
		a.firstMoment = copyGrad(grad)
		scaleGrad(a.firstMoment, 1-a.decayRate(1))
	} else {
		decayRate := a.decayRate(1)
		scaleGrad(a.firstMoment, decayRate)

		keepRate := 1 - decayRate
		for variable, vec := range grad {
			momentVec := a.firstMoment[variable]
			v := vec.Copy()
			v.Scale(vec.Creator().MakeNumeric(keepRate))
			momentVec.Add(v)
		}
	}

	if a.secondMoment == nil {
		a.secondMoment = copyGrad(grad)
		for _, v := range a.secondMoment {
			anyvec.Pow(v, v.Creator().MakeNumeric(2))
		}
		scaleGrad(a.secondMoment, 1-a.decayRate(2))
	} else {
		decayRate := a.decayRate(2)
		scaleGrad(a.secondMoment, decayRate)
		keepRate := 1 - decayRate
		for variable, vec := range grad {
			momentVec := a.secondMoment[variable]
			v := vec.Copy()
			anyvec.Pow(v, v.Creator().MakeNumeric(2))
			v.Scale(v.Creator().MakeNumeric(keepRate))
			momentVec.Add(v)
		}
	}
}

func (a *Adam) decayRate(moment int) float64 {
	if moment == 1 {
		return valueOrDefault(a.DecayRate1, adamDefaultDecayRate1)
	} else if moment == 2 {
		return valueOrDefault(a.DecayRate2, adamDefaultDecayRate2)
	} else {
		panic("invalid moment.")
	}
}

func (a *Adam) damping() float64 {
	if a.Damping != 0 {
		return a.Damping
	} else {
		return adamDefaultDamping
	}
}
