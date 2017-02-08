package anynet

import "github.com/unixpickle/anydiff"

// A Mixer combines batches of inputs from two different
// sources into a single vector.
type Mixer interface {
	Mix(in1, in2 anydiff.Res, batch int) anydiff.Res
}

// An AddMixer combines two inputs by applying layers to
// each of them, adding the results together, and then
// applying an output layer to the sum.
type AddMixer struct {
	In1      Layer
	In2      Layer
	OutTrans Layer
}

// Mix applies a.In1 to in1 and a.In2 to in2, then adds
// the results, then applies a.OutTrans.
func (a *AddMixer) Mix(in1, in2 anydiff.Res, batch int) anydiff.Res {
	return a.OutTrans.Apply(anydiff.Add(
		a.In1.Apply(in1, batch),
		a.In2.Apply(in2, batch),
	), batch)
}
