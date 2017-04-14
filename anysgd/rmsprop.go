package anysgd

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
)

const (
	rmspropDefaultDecayRate = 0.9
	rmspropDefaultDamping   = 1e-8
)

// RMSProp implements the RMSProp regularizer; see:
// http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf.
type RMSProp struct {
	// The decay rate for the running average.
	// If it is 0, a default of 0.9 is used.
	DecayRate float64

	// Damping is used to prevent divisions by zero.
	// This should be very small.
	// If it is 0, a default is used.
	Damping float64

	moment anydiff.Grad
}

// Transform transforms the gradient using RMSProp.
//
// This is not thread-safe.
func (r *RMSProp) Transform(realGrad anydiff.Grad) anydiff.Grad {
	if r.moment == nil {
		r.moment = anydiff.Grad{}
		for v, grad := range realGrad {
			sq := grad.Copy()
			anyvec.Pow(sq, sq.Creator().MakeNumeric(2))
			r.moment[v] = sq
		}
	} else {
		for v, grad := range realGrad {
			sq := grad.Copy()
			anyvec.Pow(sq, sq.Creator().MakeNumeric(2))
			sq.Sub(r.moment[v])
			sq.Scale(sq.Creator().MakeNumeric(1 - r.decayRate()))
			r.moment[v].Add(sq)
		}
	}
	for v, grad := range realGrad {
		div := r.moment[v].Copy()
		div.AddScalar(div.Creator().MakeNumeric(r.damping()))
		anyvec.Pow(div, div.Creator().MakeNumeric(-0.5))
		grad.Mul(div)
	}
	return realGrad
}

func (r *RMSProp) decayRate() float64 {
	if r.DecayRate == 0 {
		return rmspropDefaultDecayRate
	} else {
		return r.DecayRate
	}
}

func (r *RMSProp) damping() float64 {
	if r.Damping == 0 {
		return rmspropDefaultDamping
	} else {
		return r.Damping
	}
}
