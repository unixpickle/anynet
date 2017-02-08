package anynet

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func init() {
	var a AddMixer
	serializer.RegisterTypedDeserializer(a.SerializerType(), DeserializeAddMixer)
}

// A Mixer combines batches of inputs from two different
// sources into a single vector.
type Mixer interface {
	Mix(in1, in2 anydiff.Res, batch int) anydiff.Res
}

// An AddMixer combines two inputs by applying layers to
// each of them, adding the results together, and then
// applying an output layer to the sum.
type AddMixer struct {
	In1 Layer
	In2 Layer
	Out Layer
}

// DeserializeAddMixer deserializes an AddMixer.
func DeserializeAddMixer(d []byte) (*AddMixer, error) {
	var res AddMixer
	if err := serializer.DeserializeAny(d, &res.In1, &res.In2, &res.Out); err != nil {
		return nil, essentials.AddCtx("deserialize AddMixer", err)
	}
	return &res, nil
}

// Mix applies a.In1 to in1 and a.In2 to in2, then adds
// the results, then applies a.Out.
func (a *AddMixer) Mix(in1, in2 anydiff.Res, batch int) anydiff.Res {
	return a.Out.Apply(anydiff.Add(
		a.In1.Apply(in1, batch),
		a.In2.Apply(in2, batch),
	), batch)
}

// Parameters gets the parameters of all the layers that
// implement Parameterizer.
func (a *AddMixer) Parameters() []*anydiff.Var {
	var res []*anydiff.Var
	for _, v := range []Layer{a.In1, a.In2, a.Out} {
		if p, ok := v.(Parameterizer); ok {
			res = append(res, p.Parameters()...)
		}
	}
	return res
}

// SerializerType returns the unique ID used to serialize
// an AddMixer with the serializer package.
func (a *AddMixer) SerializerType() string {
	return "github.com/unixpickle/anynet.AddMixer"
}

// Serialize attempts to serialize the AddMixer.
func (a *AddMixer) Serialize() ([]byte, error) {
	return serializer.SerializeAny(a.In1, a.In2, a.Out)
}
