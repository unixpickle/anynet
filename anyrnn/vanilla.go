package anyrnn

import (
	"errors"
	"math"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvecsave"
	"github.com/unixpickle/serializer"
)

func init() {
	var v Vanilla
	serializer.RegisterTypedDeserializer(v.SerializerType(), DeserializeVanilla)
}

// Vanilla implements a standard RNN block with a single
// linear transformation and squashing function.
//
// The output (or state) of a vanilla RNN is given as
//
//     out := s(Ws*inState + Wi*input + b)
//
// Where s is some squashing function, Ws is the state
// transformation, Wi is the input transformation, and b
// is a bias.
type Vanilla struct {
	InCount  int
	OutCount int

	StateWeights *anydiff.Var
	InputWeights *anydiff.Var
	Biases       *anydiff.Var
	StartState   *anydiff.Var
	Activation   anynet.Layer
}

// DeserializeVanilla deserializes a Vanilla block.
func DeserializeVanilla(d []byte) (*Vanilla, error) {
	var stW, inW, b, start *anyvecsave.S
	var a anynet.Layer
	if err := serializer.DeserializeAny(d, &stW, &inW, &b, &start, &a); err != nil {
		return nil, err
	}

	outCount := b.Vector.Len()
	inCount := inW.Vector.Len() / outCount

	if stW.Vector.Len() != outCount*outCount {
		return nil, errors.New("incorrect vanilla state matrix size")
	}
	if inW.Vector.Len() != inCount*outCount {
		return nil, errors.New("incorrect vanilla input matrix size")
	}
	if start.Vector.Len() != outCount {
		return nil, errors.New("incorrect vanilla start state size")
	}

	return &Vanilla{
		InCount:      inCount,
		OutCount:     outCount,
		StateWeights: anydiff.NewVar(stW.Vector),
		InputWeights: anydiff.NewVar(inW.Vector),
		Biases:       anydiff.NewVar(b.Vector),
		StartState:   anydiff.NewVar(start.Vector),
		Activation:   a,
	}, nil
}

// NewVanilla creates a new, randomized Vanilla block.
func NewVanilla(c anyvec.Creator, in, out int, activation anynet.Layer) *Vanilla {
	res := NewVanillaZero(c, in, out, activation)

	anyvec.Rand(res.StateWeights.Vector, anyvec.Normal, nil)
	anyvec.Rand(res.InputWeights.Vector, anyvec.Normal, nil)
	res.StateWeights.Vector.Scale(1 / math.Sqrt(float64(out)))
	res.InputWeights.Vector.Scale(1 / math.Sqrt(float64(in)))

	return res
}

// NewVanillaZero creates a new, zero'd out Vanilla block.
func NewVanillaZero(c anyvec.Creator, in, out int, activation anynet.Layer) *Vanilla {
	return &Vanilla{
		InCount:      in,
		OutCount:     out,
		StateWeights: anydiff.NewVar(c.MakeVector(out * out)),
		InputWeights: anydiff.NewVar(c.MakeVector(in * out)),
		Biases:       anydiff.NewVar(c.MakeVector(out)),
		StartState:   anydiff.NewVar(c.MakeVector(out)),
		Activation:   activation,
	}
}

// State generates an initial state.
func (v *Vanilla) State(n int) State {
	return NewVecState(v.StartState.Vector, n)
}

// PropagateStart propagates through the start state.
func (v *Vanilla) PropagateStart(s StateGrad, g anydiff.Grad) {
	s.(*VecState).PropagateStart(v.StartState, g)
}

// Step performs one timestep.
func (v *Vanilla) Step(s State, in anyvec.Vector) Res {
	res := &vanillaRes{
		InPool:    anydiff.NewVar(in),
		StatePool: anydiff.NewVar(s.(*VecState).Vector),
		V:         anydiff.VarSet{},
	}
	res.V.Add(v.StartState)

	wState := applyWeights(v.OutCount, v.OutCount, v.StateWeights, res.StatePool)
	wInput := applyWeights(v.InCount, v.OutCount, v.InputWeights, res.InPool)
	sum := anydiff.Add(wState, wInput)
	biased := anydiff.AddRepeated(sum, v.Biases)
	res.Out = v.Activation.Apply(biased, s.Present().NumPresent())
	res.OutState = &VecState{Vector: res.Out.Output(), PresentMap: s.Present()}
	res.V = anydiff.MergeVarSets(res.V, res.Out.Vars())

	return res
}

// Parameters returns all of the block's parameters,
// including those of v.Activation if it has any.
func (v *Vanilla) Parameters() []*anydiff.Var {
	res := []*anydiff.Var{v.StateWeights, v.InputWeights, v.Biases, v.StartState}
	if p, ok := v.Activation.(anynet.Parameterizer); ok {
		res = append(res, p.Parameters()...)
	}
	return res
}

// SerializerType returns the unique ID used to serialize
// a Vanilla with the serializer package.
func (v *Vanilla) SerializerType() string {
	return "github.com/unixpickle/anynet/anyrnn.Vanilla"
}

// Serialize serializes the Vanilla.
func (v *Vanilla) Serialize() ([]byte, error) {
	stW := &anyvecsave.S{Vector: v.StateWeights.Vector}
	inW := &anyvecsave.S{Vector: v.InputWeights.Vector}
	b := &anyvecsave.S{Vector: v.Biases.Vector}
	start := &anyvecsave.S{Vector: v.StartState.Vector}
	return serializer.SerializeAny(stW, inW, b, start, v.Activation)
}

type vanillaRes struct {
	InPool    *anydiff.Var
	StatePool *anydiff.Var
	OutState  State
	Out       anydiff.Res
	V         anydiff.VarSet
}

func (v *vanillaRes) State() State {
	return v.OutState
}

func (v *vanillaRes) Output() anyvec.Vector {
	return v.Out.Output()
}

func (v *vanillaRes) Vars() anydiff.VarSet {
	return v.V
}

func (v *vanillaRes) Propagate(u anyvec.Vector, s StateGrad, g anydiff.Grad) (anyvec.Vector,
	StateGrad) {
	down := v.InPool.Vector.Creator().MakeVector(v.InPool.Vector.Len())
	downState := v.StatePool.Vector.Creator().MakeVector(v.StatePool.Vector.Len())
	g[v.InPool] = down
	g[v.StatePool] = downState
	if s != nil {
		u.Add(s.(*VecState).Vector)
	}
	v.Out.Propagate(u, g)
	delete(g, v.InPool)
	delete(g, v.StatePool)
	return down, &VecState{
		Vector:     downState,
		PresentMap: v.OutState.Present(),
	}
}

func applyWeights(in, out int, weights anydiff.Res, batch anydiff.Res) anydiff.Res {
	weightMat := &anydiff.Matrix{Data: weights, Rows: out, Cols: in}
	inMat := &anydiff.Matrix{Data: batch, Rows: batch.Output().Len() / in, Cols: in}
	return anydiff.MatMul(false, true, inMat, weightMat).Data
}
