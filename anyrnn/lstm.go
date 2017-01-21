package anyrnn

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvecsave"
	"github.com/unixpickle/serializer"
)

const lstmRememberBias = 1

func init() {
	var l LSTMGate
	serializer.RegisterTypedDeserializer(l.SerializerType(), DeserializeLSTMGate)
	var lstm LSTM
	serializer.RegisterTypedDeserializer(lstm.SerializerType(), DeserializeLSTM)
}

// LSTM is a long short-term memory block.
type LSTM struct {
	InValue   *LSTMGate
	In        *LSTMGate
	Remember  *LSTMGate
	Output    *LSTMGate
	InitState *anydiff.Var
}

// DeserializeLSTM deserializes an LSTM.
func DeserializeLSTM(d []byte) (*LSTM, error) {
	var inVal, in, rem, out *LSTMGate
	var initState *anyvecsave.S
	if err := serializer.DeserializeAny(d, &inVal, &in, &rem, &out, &initState); err != nil {
		return nil, err
	}
	return &LSTM{
		InValue:   inVal,
		In:        in,
		Remember:  rem,
		Output:    out,
		InitState: anydiff.NewVar(initState.Vector),
	}, nil
}

// NewLSTM creates a new, randomized LSTM.
//
// The remember gates of the LSTM are initially biased to
// remember things.
func NewLSTM(c anyvec.Creator, in, state int) *LSTM {
	res := &LSTM{
		InValue:   NewLSTMGate(c, in, state, anynet.Tanh),
		In:        NewLSTMGate(c, in, state, anynet.Sigmoid),
		Remember:  NewLSTMGate(c, in, state, anynet.Sigmoid),
		Output:    NewLSTMGate(c, in, state, anynet.Sigmoid),
		InitState: anydiff.NewVar(c.MakeVector(state)),
	}
	res.Remember.Biases.Vector.AddScaler(c.MakeNumeric(lstmRememberBias))
	return res
}

// NewLSTMZero creates a zero'd LSTM.
func NewLSTMZero(c anyvec.Creator, in, state int) *LSTM {
	return &LSTM{
		InValue:   NewLSTMGateZero(c, in, state, anynet.Tanh),
		In:        NewLSTMGateZero(c, in, state, anynet.Sigmoid),
		Remember:  NewLSTMGateZero(c, in, state, anynet.Sigmoid),
		Output:    NewLSTMGateZero(c, in, state, anynet.Sigmoid),
		InitState: anydiff.NewVar(c.MakeVector(state)),
	}
}

// Parameters returns the parameters of the block.
func (l *LSTM) Parameters() []*anydiff.Var {
	res := []*anydiff.Var{l.InitState}
	for _, g := range []*LSTMGate{l.InValue, l.In, l.Remember, l.Output} {
		res = append(res, g.Parameters()...)
	}
	return res
}

// SerializerType returns the unique ID used to serialize
// an LSTM with the serializer package.
func (l *LSTM) SerializerType() string {
	return "github.com/unixpickle/anynet/anyrnn.LSTM"
}

// Serialize serializes the LSTM.
func (l *LSTM) Serialize() ([]byte, error) {
	return serializer.SerializeAny(l.InValue, l.In, l.Remember, l.Output,
		&anyvecsave.S{Vector: l.InitState.Vector})
}

// An LSTMGate computes a value based on the previous
// output, the current state, and the input.
type LSTMGate struct {
	StateWeights *anydiff.Var
	InputWeights *anydiff.Var
	Peephole     *anydiff.Var
	Biases       *anydiff.Var
	Activation   anynet.Layer
}

// DeserializeLSTMGate deserializes an LSTMGate.
func DeserializeLSTMGate(d []byte) (*LSTMGate, error) {
	var sw, iw, p, b *anyvecsave.S
	var a anynet.Activation
	if err := serializer.DeserializeAny(d, &sw, &iw, &p, &b, &a); err != nil {
		return nil, err
	}
	return &LSTMGate{
		StateWeights: anydiff.NewVar(sw.Vector),
		InputWeights: anydiff.NewVar(iw.Vector),
		Peephole:     anydiff.NewVar(p.Vector),
		Biases:       anydiff.NewVar(b.Vector),
		Activation:   a,
	}, nil
}

// NewLSTMGate creates a randomized LSTM gate.
func NewLSTMGate(c anyvec.Creator, in, state int, activation anynet.Layer) *LSTMGate {
	// Hijack the vanilla randomization code.
	vn := NewVanilla(c, in, state, activation)
	return &LSTMGate{
		StateWeights: vn.StateWeights,
		InputWeights: vn.InputWeights,
		Peephole:     anydiff.NewVar(c.MakeVector(state)),
		Biases:       vn.Biases,
		Activation:   activation,
	}
}

// NewLSTMGateZero creates a zero'd LSTM gate.
func NewLSTMGateZero(c anyvec.Creator, in, state int, activation anynet.Layer) *LSTMGate {
	return &LSTMGate{
		StateWeights: anydiff.NewVar(c.MakeVector(state * state)),
		InputWeights: anydiff.NewVar(c.MakeVector(state * in)),
		Peephole:     anydiff.NewVar(c.MakeVector(state)),
		Biases:       anydiff.NewVar(c.MakeVector(state)),
		Activation:   activation,
	}
}

// Parameters returns the parameters of the gate.
func (l *LSTMGate) Parameters() []*anydiff.Var {
	return []*anydiff.Var{l.StateWeights, l.InputWeights, l.Peephole, l.Biases}
}

// SerializerType returns the unique ID used to serialize
// an LSTM gate with the serializer package.
func (l *LSTMGate) SerializerType() string {
	return "github.com/unixpickle/anynet/anyrnn.LSTMGate"
}

// Serialize serializes the gate.
func (l *LSTMGate) Serialize() ([]byte, error) {
	sw := &anyvecsave.S{Vector: l.StateWeights.Vector}
	iw := &anyvecsave.S{Vector: l.InputWeights.Vector}
	p := &anyvecsave.S{Vector: l.Peephole.Vector}
	b := &anyvecsave.S{Vector: l.Biases.Vector}
	return serializer.SerializeAny(sw, iw, p, b, l.Activation)
}
