package anyrnn

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func init() {
	var b Bidir
	serializer.RegisterTypedDeserializer(b.SerializerType(), DeserializeBidir)
}

// Bidir implements a bi-directional RNN.
//
// In a bi-directional RNN, a forward block is evaluated
// on the input sequence, while a backward block is mapped
// over the reversed input sequence.
// Then, outputs from the forward and backward block for
// corresponding timesteps in the original sequence are
// combined using the mixer.
//
// The first input to the Mixer is from the forward block;
// the second is from the backward block.
type Bidir struct {
	Forward  Block
	Backward Block
	Mixer    anynet.Mixer
}

// DeserializeBidir deserializes a Bidir.
func DeserializeBidir(d []byte) (*Bidir, error) {
	var res Bidir
	err := serializer.DeserializeAny(d, &res.Forward, &res.Backward, &res.Mixer)
	if err != nil {
		return nil, essentials.AddCtx("deserialize Bidir", err)
	}
	return &res, nil
}

// Apply applies the bidirectional RNN.
func (b *Bidir) Apply(in anyseq.Seq) anyseq.Seq {
	return anyseq.Pool(in, func(in anyseq.Seq) anyseq.Seq {
		forwOut := Map(in, b.Forward)
		backOut := anyseq.Reverse(Map(anyseq.Reverse(in), b.Backward))
		return anyseq.MapN(func(n int, v ...anydiff.Res) anydiff.Res {
			return b.Mixer.Mix(v[0], v[1], n)
		}, forwOut, backOut)
	})
}

// Parameters returns the parameters of the blocks and
// Mixer if they implement anynet.Parameterizer.
func (b *Bidir) Parameters() []*anydiff.Var {
	var res []*anydiff.Var
	for _, x := range []interface{}{b.Forward, b.Backward, b.Mixer} {
		if p, ok := x.(anynet.Parameterizer); ok {
			res = append(res, p.Parameters()...)
		}
	}
	return res
}

// SerializerType returns the unique ID used to serialize
// a Bidir with the serializer package.
func (b *Bidir) SerializerType() string {
	return "github.com/unixpickle/anynet/anyrnn.Bidir"
}

// Serialize serializes the Bidir.
func (b *Bidir) Serialize() ([]byte, error) {
	return serializer.SerializeAny(b.Forward, b.Backward, b.Mixer)
}
