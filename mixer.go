package anynet

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func init() {
	var a AddMixer
	serializer.RegisterTypedDeserializer(a.SerializerType(), DeserializeAddMixer)
	var c ConcatMixer
	serializer.RegisterTypedDeserializer(c.SerializerType(), DeserializeConcatMixer)
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
	return AllParameters(a.In1, a.In2, a.Out)
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

// A ConcatMixer mixes inputs by concatenating inputs.
type ConcatMixer struct{}

// DeserializeConcatMixer deserializes a ConcatMixer.
func DeserializeConcatMixer(d []byte) (ConcatMixer, error) {
	return ConcatMixer{}, nil
}

// Mix produces a vector of concatenated vectors, like
// [in1[0], in2[0], in1[1], in2[1], ...], where in1[n]
// represents the n-th vector in the batch represented
// by in1.
func (c ConcatMixer) Mix(in1, in2 anydiff.Res, batch int) anydiff.Res {
	return anydiff.Pool(in1, func(in1 anydiff.Res) anydiff.Res {
		return anydiff.Pool(in2, func(in2 anydiff.Res) anydiff.Res {
			var res []anydiff.Res
			v1Len := in1.Output().Len() / batch
			v2Len := in2.Output().Len() / batch
			for i := 0; i < batch; i++ {
				res = append(res, anydiff.Slice(in1, i*v1Len, (i+1)*v1Len),
					anydiff.Slice(in2, i*v2Len, (i+1)*v2Len))
			}
			return anydiff.Concat(res...)
		})
	})
}

// SerializerType returns the unique ID used to serialize
// a ConcatMixer with the serializer package.
func (c ConcatMixer) SerializerType() string {
	return "github.com/unixpickle/anynet.ConcatMixer"
}

// Serialize serializes the instance.
func (c ConcatMixer) Serialize() ([]byte, error) {
	return []byte{}, nil
}
