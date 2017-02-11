package anyrnn

import (
	"errors"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func init() {
	var l LayerBlock
	serializer.RegisterTypedDeserializer(l.SerializerType(), DeserializeLayerBlock)
}

// A LayerBlock is a stateless Block that applies a
// feed-forward neural network (or a layer thereof) to its
// inputs.
type LayerBlock struct {
	Layer anynet.Layer
}

// DeserializeLayerBlock deserializes a LayerBlock.
func DeserializeLayerBlock(d []byte) (*LayerBlock, error) {
	n, err := anynet.DeserializeNet(d)
	if err != nil {
		return nil, essentials.AddCtx("deserialize LayerBlock", err)
	}
	if len(n) != 1 {
		return nil, errors.New("deserialize LayerBlock: multiple Layers")
	}
	return &LayerBlock{Layer: n[0]}, nil
}

// Start creates an empty start state.
func (l *LayerBlock) Start(n int) State {
	pres := make([]bool, n)
	for i := range pres {
		pres[i] = true
	}
	return &emptyState{P: pres}
}

// PropagateStart does nothing, since the block is
// state-less.
func (l *LayerBlock) PropagateStart(s StateGrad, g anydiff.Grad) {
}

// Step applies the block for a single timestep.
func (l *LayerBlock) Step(s State, in anyvec.Vector) Res {
	p := anydiff.NewVar(in)
	out := l.Layer.Apply(p, s.Present().NumPresent())
	v := anydiff.MergeVarSets(out.Vars())
	v.Del(p)
	return &layerBlockRes{
		S:    s.(*emptyState),
		Pool: p,
		Res:  out,
		V:    v,
	}
}

// Parameters returns the parameters of the layer if it is
// an anynet.Parameterizer.
func (l *LayerBlock) Parameters() []*anydiff.Var {
	return anynet.Net{l.Layer}.Parameters()
}

// SerializerType returns the unique ID used to serialize
// a LayerBlock with the serializer package.
func (l *LayerBlock) SerializerType() string {
	return "github.com/unixpickle/anynet/anyrnn.LayerBlock"
}

// Serialize serializes the block if the Layer can be
// serialized.
func (l *LayerBlock) Serialize() ([]byte, error) {
	return anynet.Net{l.Layer}.Serialize()
}

type layerBlockRes struct {
	S    *emptyState
	Pool *anydiff.Var
	Res  anydiff.Res
	V    anydiff.VarSet
}

func (l *layerBlockRes) State() State {
	return l.S
}

func (l *layerBlockRes) Output() anyvec.Vector {
	return l.Res.Output()
}

func (l *layerBlockRes) Vars() anydiff.VarSet {
	return l.V
}

func (l *layerBlockRes) Propagate(u anyvec.Vector, s StateGrad, g anydiff.Grad) (anyvec.Vector,
	StateGrad) {
	inDown := l.Pool.Vector.Creator().MakeVector(l.Pool.Vector.Len())
	g[l.Pool] = inDown
	l.Res.Propagate(u, g)
	delete(g, l.Pool)
	if s == nil {
		s = l.S
	}
	return inDown, s
}

type emptyState struct {
	P PresentMap
}

func (e *emptyState) Present() PresentMap {
	return e.P
}

func (e *emptyState) Reduce(p PresentMap) State {
	return &emptyState{P: p}
}

func (e *emptyState) Expand(p PresentMap) StateGrad {
	return &emptyState{P: p}
}
