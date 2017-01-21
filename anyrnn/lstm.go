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
		return nil, err
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

// Start returns the start state for the RNN.
func (l *LSTM) Start(n int) State {
	return &lstmState{
		lastOut:  NewVecState(l.InitLastOut.Output(), n),
		internal: NewVecState(l.InitInternal.Output(), n),
	}
}

// PropagateStart propagates through the start state.
func (l *LSTM) PropagateStart(s StateGrad, g anydiff.Grad) {
	ls := s.(*lstmState)
	ls.lastOut.PropagateStart(l.InitLastOut, g)
	ls.internal.PropagateStart(l.InitInternal, g)
}

// Step performs one timestep.
func (l *LSTM) Step(s State, in anyvec.Vector) Res {
	ls := s.(*lstmState)

	res := &lstmRes{
		V:                anydiff.VarSet{},
		InPool:           anydiff.NewVar(in),
		LastOutPool:      anydiff.NewVar(ls.lastOut.Vector),
		LastInternalPool: anydiff.NewVar(ls.internal.Vector),
	}

	for _, p := range l.Parameters() {
		res.V.Add(p)
	}

	inVal := l.InValue.Apply(res.LastOutPool, res.InPool, res.LastInternalPool)
	inGate := l.In.Apply(res.LastOutPool, res.InPool, res.LastInternalPool)
	remGate := l.Remember.Apply(res.LastOutPool, res.InPool, res.LastInternalPool)

	res.InternalRes = anydiff.Add(
		anydiff.Scale(inVal, inGate),
		anydiff.Scale(res.LastInternalPool, remGate),
	)
	res.InternalPool = anydiff.NewVar(res.InternalRes.Output())

	// Peephole of output gate gets to see the new internals.
	outGate := l.Output.Apply(res.LastOutPool, res.InPool, res.InternalPool)
	squashedOut := l.OutSquash.Apply(res.InternalPool, s.Present().NumPresent())

	res.OutputRes = anydiff.Mul(outGate, squashedOut)
	res.OutState = &lstmState{
		lastOut: &VecState{
			Vector:     res.OutputRes.Output(),
			PresentMap: s.Present(),
		},
		internal: &VecState{
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

type lstmState struct {
	lastOut  *VecState
	internal *VecState
}

func (l *lstmState) Present() PresentMap {
	return l.lastOut.Present()
}

func (l *lstmState) Reduce(p PresentMap) State {
	return &lstmState{
		lastOut:  l.lastOut.Reduce(p).(*VecState),
		internal: l.internal.Reduce(p).(*VecState),
	}
}

func (l *lstmState) Expand(p PresentMap) StateGrad {
	return &lstmState{
		lastOut:  l.lastOut.Expand(p).(*VecState),
		internal: l.internal.Expand(p).(*VecState),
	}
}

type lstmRes struct {
	OutState *lstmState
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
		u.Add(s.(*lstmState).lastOut.Vector)
	}
	l.OutputRes.Propagate(u, g)
	internalUpstream := g[l.InternalPool]
	delete(g, l.InternalPool)

	if s != nil {
		internalUpstream.Add(s.(*lstmState).internal.Vector)
	}
	l.InternalRes.Propagate(internalUpstream, g)

	downState := &lstmState{
		lastOut: &VecState{
			Vector:     g[l.LastOutPool],
			PresentMap: l.OutState.Present(),
		},
		internal: &VecState{
			Vector:     g[l.InternalPool],
			PresentMap: l.OutState.Present(),
		},
	}
	inputDown := g[l.InPool]

	return inputDown, downState
}

func (l *lstmRes) pools() []*anydiff.Var {
	return []*anydiff.Var{l.InPool, l.LastOutPool, l.LastInternalPool, l.InternalPool}
}
