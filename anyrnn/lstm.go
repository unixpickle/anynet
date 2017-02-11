package anyrnn

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvecsave"
	"github.com/unixpickle/essentials"
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
	InValue      *LSTMGate
	In           *LSTMGate
	Remember     *LSTMGate
	Output       *LSTMGate
	OutSquash    anynet.Layer
	InitLastOut  *anydiff.Var
	InitInternal *anydiff.Var
}

// DeserializeLSTM deserializes an LSTM.
func DeserializeLSTM(d []byte) (*LSTM, error) {
	var inVal, in, rem, out *LSTMGate
	var outSquash anynet.Layer
	var initLast, initInt *anyvecsave.S
	err := serializer.DeserializeAny(d, &inVal, &in, &rem, &out, &outSquash,
		&initLast, &initInt)
	if err != nil {
		return nil, essentials.AddCtx("deserialize LSTM", err)
	}
	return &LSTM{
		InValue:      inVal,
		In:           in,
		Remember:     rem,
		Output:       out,
		OutSquash:    outSquash,
		InitLastOut:  anydiff.NewVar(initLast.Vector),
		InitInternal: anydiff.NewVar(initInt.Vector),
	}, nil
}

// NewLSTM creates a new, randomized LSTM.
//
// The remember gates of the LSTM are initially biased to
// remember things.
func NewLSTM(c anyvec.Creator, in, state int) *LSTM {
	res := &LSTM{
		InValue:      NewLSTMGate(c, in, state, anynet.Tanh),
		In:           NewLSTMGate(c, in, state, anynet.Sigmoid),
		Remember:     NewLSTMGate(c, in, state, anynet.Sigmoid),
		Output:       NewLSTMGate(c, in, state, anynet.Sigmoid),
		OutSquash:    anynet.Tanh,
		InitLastOut:  anydiff.NewVar(c.MakeVector(state)),
		InitInternal: anydiff.NewVar(c.MakeVector(state)),
	}
	res.Remember.Biases.Vector.AddScaler(c.MakeNumeric(lstmRememberBias))
	return res
}

// NewLSTMZero creates a zero'd LSTM.
func NewLSTMZero(c anyvec.Creator, in, state int) *LSTM {
	return &LSTM{
		InValue:      NewLSTMGateZero(c, in, state, anynet.Tanh),
		In:           NewLSTMGateZero(c, in, state, anynet.Sigmoid),
		Remember:     NewLSTMGateZero(c, in, state, anynet.Sigmoid),
		Output:       NewLSTMGateZero(c, in, state, anynet.Sigmoid),
		OutSquash:    anynet.Tanh,
		InitLastOut:  anydiff.NewVar(c.MakeVector(state)),
		InitInternal: anydiff.NewVar(c.MakeVector(state)),
	}
}

// ScaleInWeights scales the matrix entries that transform
// input values into state values.
//
// Normally, the LSTM is initialized under the assumption
// that each input component will have variance 1.
// By using ScaleInWeights, you can adjust the weights to
// deal with different scenarios (e.g. if your input is a
// one hot vector scheme).
//
// The LSTM l is returned for convenience.
func (l *LSTM) ScaleInWeights(scaler anyvec.Numeric) *LSTM {
	for _, gate := range []*LSTMGate{l.InValue, l.In, l.Remember, l.Output} {
		gate.InputWeights.Vector.Scale(scaler)
	}
	return l
}

// Start returns the start state for the RNN.
func (l *LSTM) Start(n int) State {
	return &LSTMState{
		LastOut:  NewVecState(l.InitLastOut.Output(), n),
		Internal: NewVecState(l.InitInternal.Output(), n),
	}
}

// PropagateStart propagates through the start state.
func (l *LSTM) PropagateStart(s StateGrad, g anydiff.Grad) {
	ls := s.(*LSTMState)
	ls.LastOut.PropagateStart(l.InitLastOut, g)
	ls.Internal.PropagateStart(l.InitInternal, g)
}

// Step performs one timestep.
func (l *LSTM) Step(s State, in anyvec.Vector) Res {
	ls := s.(*LSTMState)

	res := &lstmRes{
		V:                anydiff.VarSet{},
		InPool:           anydiff.NewVar(in),
		LastOutPool:      anydiff.NewVar(ls.LastOut.Vector),
		LastInternalPool: anydiff.NewVar(ls.Internal.Vector),
	}

	for _, p := range l.Parameters() {
		res.V.Add(p)
	}

	inVal := l.InValue.Apply(res.LastOutPool, res.InPool, res.LastInternalPool)
	inGate := l.In.Apply(res.LastOutPool, res.InPool, res.LastInternalPool)
	remGate := l.Remember.Apply(res.LastOutPool, res.InPool, res.LastInternalPool)

	res.InternalRes = anydiff.Add(
		anydiff.Mul(inVal, inGate),
		anydiff.Mul(res.LastInternalPool, remGate),
	)
	res.InternalPool = anydiff.NewVar(res.InternalRes.Output())

	// Peephole of output gate gets to see the new internals.
	outGate := l.Output.Apply(res.LastOutPool, res.InPool, res.InternalPool)
	squashedOut := l.OutSquash.Apply(res.InternalPool, s.Present().NumPresent())

	res.OutputRes = anydiff.Mul(outGate, squashedOut)
	res.OutState = &LSTMState{
		LastOut: &VecState{
			Vector:     res.OutputRes.Output(),
			PresentMap: s.Present(),
		},
		Internal: &VecState{
			Vector:     res.InternalRes.Output(),
			PresentMap: s.Present(),
		},
	}
	res.OutVec = res.OutputRes.Output()

	return res
}

// Parameters returns the parameters of the block.
func (l *LSTM) Parameters() []*anydiff.Var {
	res := []*anydiff.Var{l.InitLastOut, l.InitInternal}
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
	return serializer.SerializeAny(l.InValue, l.In, l.Remember, l.Output, l.OutSquash,
		&anyvecsave.S{Vector: l.InitLastOut.Vector},
		&anyvecsave.S{Vector: l.InitInternal.Vector})
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

// Apply applies the gate.
func (l *LSTMGate) Apply(state, input, internal anydiff.Res) anydiff.Res {
	outCount := l.Biases.Vector.Len()
	inCount := l.InputWeights.Vector.Len() / outCount
	weighted1 := applyWeights(outCount, outCount, l.StateWeights, state)
	weighted2 := applyWeights(inCount, outCount, l.InputWeights, input)
	peep := anydiff.ScaleRepeated(internal, l.Peephole)
	return l.Activation.Apply(anydiff.Add(
		anydiff.Add(weighted1, weighted2),
		anydiff.AddRepeated(peep, l.Biases),
	), state.Output().Len()/outCount)
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

// LSTMState is the State and StateGrad type for LSTMs.
type LSTMState struct {
	// LastOut is the last output of the block.
	LastOut *VecState

	// Internal is the last (unsquashed) internal state.
	Internal *VecState
}

// Present returns the present map.
func (l *LSTMState) Present() PresentMap {
	return l.LastOut.Present()
}

// Reduce reduces both internal states.
func (l *LSTMState) Reduce(p PresentMap) State {
	return &LSTMState{
		LastOut:  l.LastOut.Reduce(p).(*VecState),
		Internal: l.Internal.Reduce(p).(*VecState),
	}
}

// Exand expands both internal states.
func (l *LSTMState) Expand(p PresentMap) StateGrad {
	return &LSTMState{
		LastOut:  l.LastOut.Expand(p).(*VecState),
		Internal: l.Internal.Expand(p).(*VecState),
	}
}

type lstmRes struct {
	OutState *LSTMState
	OutVec   anyvec.Vector
	V        anydiff.VarSet

	InternalRes anydiff.Res
	OutputRes   anydiff.Res

	InPool           *anydiff.Var
	LastOutPool      *anydiff.Var
	LastInternalPool *anydiff.Var
	InternalPool     *anydiff.Var
}

func (l *lstmRes) State() State {
	return l.OutState
}

func (l *lstmRes) Output() anyvec.Vector {
	return l.OutVec
}

func (l *lstmRes) Vars() anydiff.VarSet {
	return l.V
}

func (l *lstmRes) Propagate(u anyvec.Vector, s StateGrad, g anydiff.Grad) (anyvec.Vector,
	StateGrad) {
	for _, p := range l.pools() {
		g[p] = p.Vector.Creator().MakeVector(p.Vector.Len())
	}
	defer func() {
		for _, p := range l.pools() {
			delete(g, p)
		}
	}()

	if s != nil {
		u.Add(s.(*LSTMState).LastOut.Vector)
	}
	l.OutputRes.Propagate(u, g)
	internalUpstream := g[l.InternalPool]
	delete(g, l.InternalPool)

	if s != nil {
		internalUpstream.Add(s.(*LSTMState).Internal.Vector)
	}
	l.InternalRes.Propagate(internalUpstream, g)

	downState := &LSTMState{
		LastOut: &VecState{
			Vector:     g[l.LastOutPool],
			PresentMap: l.OutState.Present(),
		},
		Internal: &VecState{
			Vector:     g[l.LastInternalPool],
			PresentMap: l.OutState.Present(),
		},
	}
	inputDown := g[l.InPool]

	return inputDown, downState
}

func (l *lstmRes) pools() []*anydiff.Var {
	return []*anydiff.Var{l.InPool, l.LastOutPool, l.LastInternalPool, l.InternalPool}
}
