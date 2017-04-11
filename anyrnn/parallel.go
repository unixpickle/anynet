package anyrnn

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func init() {
	serializer.RegisterTypedDeserializer((&Parallel{}).SerializerType(), DeserializeParallel)
}

// A Parallel block feeds its input to two blocks, then
// merges the blocks' outputs.
type Parallel struct {
	Block1 Block
	Block2 Block
	Mixer  anynet.Mixer
}

// DeserializeParallel deserializes a Parallel.
func DeserializeParallel(d []byte) (*Parallel, error) {
	var res Parallel
	err := serializer.DeserializeAny(d, &res.Block1, &res.Block2, &res.Mixer)
	if err != nil {
		return nil, essentials.AddCtx("deserialize Parallel", err)
	}
	return &res, nil
}

// Start produces a start state.
func (p *Parallel) Start(n int) State {
	return &ParallelState{
		State1: p.Block1.Start(n),
		State2: p.Block2.Start(n),
	}
}

// PropagateStart back-propagates through the start state.
func (p *Parallel) PropagateStart(s StateGrad, g anydiff.Grad) {
	sg := s.(*ParallelGrad)
	p.Block1.PropagateStart(sg.Grad1, g)
	p.Block2.PropagateStart(sg.Grad2, g)
}

// Step takes a timestep.
func (p *Parallel) Step(s State, in anyvec.Vector) Res {
	state := s.(*ParallelState)
	res := &parallelRes{
		Res1: p.Block1.Step(state.State1, in),
		Res2: p.Block2.Step(state.State2, in),
	}
	res.V = anydiff.MergeVarSets(res.Res1.Vars(), res.Res2.Vars())
	res.Pool1 = anydiff.NewVar(res.Res1.Output())
	res.Pool2 = anydiff.NewVar(res.Res2.Output())
	res.OutRes = p.Mixer.Mix(res.Pool1, res.Pool2, s.Present().NumPresent())
	res.OutState = &ParallelState{State1: res.Res1.State(), State2: res.Res2.State()}
	return res
}

// Parameters returns the parameters of the block, which
// are taken from block 1, block 2, and the mixer in that
// order.
func (p *Parallel) Parameters() []*anydiff.Var {
	return anynet.AllParameters(p.Block1, p.Block2, p.Mixer)
}

// SerializerType returns the unique ID used to serialize
// a Parallel with the serializer package.
func (p *Parallel) SerializerType() string {
	return "github.com/unixpickle/anynet/anyrnn.Parallel"
}

// Serialize serializes the block.
func (p *Parallel) Serialize() ([]byte, error) {
	return serializer.SerializeAny(p.Block1, p.Block2, p.Mixer)
}

// ParallelState stores the state of a Parallel block.
type ParallelState struct {
	State1 State
	State2 State
}

// Present returns the present map of one of the internal
// states.
func (p *ParallelState) Present() PresentMap {
	return p.State1.Present()
}

// Reduce reduces the internal states.
func (p *ParallelState) Reduce(pres PresentMap) State {
	return &ParallelState{
		State1: p.State1.Reduce(pres),
		State2: p.State2.Reduce(pres),
	}
}

// ParallelGrad stores the state gradient of a Parallel
// block.
type ParallelGrad struct {
	Grad1 StateGrad
	Grad2 StateGrad
}

// Present returns the present map of one of the internal
// state grads.
func (p *ParallelGrad) Present() PresentMap {
	return p.Grad1.Present()
}

// Expand expands all the internal state grads.
func (p *ParallelGrad) Expand(pres PresentMap) StateGrad {
	return &ParallelGrad{
		Grad1: p.Grad1.Expand(pres),
		Grad2: p.Grad2.Expand(pres),
	}
}

type parallelRes struct {
	Res1     Res
	Res2     Res
	OutRes   anydiff.Res
	Pool1    *anydiff.Var
	Pool2    *anydiff.Var
	OutState *ParallelState
	V        anydiff.VarSet
}

func (p *parallelRes) State() State {
	return p.OutState
}

func (p *parallelRes) Output() anyvec.Vector {
	return p.OutRes.Output()
}

func (p *parallelRes) Vars() anydiff.VarSet {
	return p.V
}

func (p *parallelRes) Propagate(u anyvec.Vector, s StateGrad,
	g anydiff.Grad) (anyvec.Vector, StateGrad) {
	for _, p := range []*anydiff.Var{p.Pool1, p.Pool2} {
		g[p] = p.Vector.Creator().MakeVector(p.Vector.Len())
		defer func(p *anydiff.Var) {
			delete(g, p)
		}(p)
	}
	p.OutRes.Propagate(u, g)
	var sg1, sg2 StateGrad
	if s != nil {
		sg := s.(*ParallelGrad)
		sg1, sg2 = sg.Grad1, sg.Grad2
	}
	inGrad1, downGrad1 := p.Res1.Propagate(g[p.Pool1], sg1, g)
	inGrad2, downGrad2 := p.Res2.Propagate(g[p.Pool2], sg2, g)
	inGrad1.Add(inGrad2)
	return inGrad1, &ParallelGrad{Grad1: downGrad1, Grad2: downGrad2}
}
