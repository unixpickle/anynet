package anyrnn

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
)

// A FuncBlock is a Block which applies a function to
// transform a state-input pair into a state-output pair.
type FuncBlock struct {
	// Func applies the block.
	Func func(in, state anydiff.Res, batch int) (out, newState anydiff.Res)

	// MakeStart produces the initial state vector.
	MakeStart func(n int) anydiff.Res
}

// Start generates an initial *FuncBlockState.
func (f *FuncBlock) Start(n int) State {
	r := f.MakeStart(n)
	pm := make(PresentMap, n)
	for i := range pm {
		pm[i] = true
	}
	return &FuncBlockState{
		VecState: &VecState{
			Vector:     r.Output(),
			PresentMap: pm,
		},
		StartRes: r,
		V:        r.Vars(),
	}
}

// PropagateStart back-propagates through the start state.
func (f *FuncBlock) PropagateStart(s StateGrad, g anydiff.Grad) {
	fs := s.(*FuncBlockState)
	fs.StartRes.Propagate(fs.VecState.Vector, g)
}

// Step applies the block for a timestep.
func (f *FuncBlock) Step(s State, in anyvec.Vector) Res {
	fs := s.(*FuncBlockState)
	inPool := anydiff.NewVar(in)
	statePool := anydiff.NewVar(fs.Vector)
	out, state := f.Func(inPool, statePool, s.Present().NumPresent())
	stateVars := anydiff.MergeVarSets(fs.V, state.Vars())
	allVars := anydiff.MergeVarSets(stateVars, out.Vars())
	for _, x := range []anydiff.VarSet{stateVars, allVars} {
		x.Del(inPool)
		x.Del(statePool)
	}
	newState := &FuncBlockState{
		VecState: &VecState{
			PresentMap: fs.PresentMap,
			Vector:     state.Output(),
		},
		V:        stateVars,
		StartRes: fs.StartRes,
	}
	return &funcBlockRes{
		InPool:    inPool,
		StatePool: statePool,
		OutRes:    out,
		StateRes:  state,
		OutState:  newState,
		V:         allVars,
	}
}

// FuncBlockState is the State and StateGrad type used by
// FuncBlock.
type FuncBlockState struct {
	*VecState
	V        anydiff.VarSet
	StartRes anydiff.Res
}

// Reduce reduces the state to the given sequences.
func (f *FuncBlockState) Reduce(p PresentMap) State {
	return &FuncBlockState{
		VecState: f.VecState.Reduce(p).(*VecState),
		V:        f.V,
		StartRes: f.StartRes,
	}
}

// Expand expands the state.
func (f *FuncBlockState) Expand(p PresentMap) StateGrad {
	return &FuncBlockState{
		VecState: f.VecState.Expand(p).(*VecState),
		V:        f.V,
		StartRes: f.StartRes,
	}
}

type funcBlockRes struct {
	InPool    *anydiff.Var
	StatePool *anydiff.Var
	OutRes    anydiff.Res
	StateRes  anydiff.Res
	OutState  *FuncBlockState
	V         anydiff.VarSet
}

func (f *funcBlockRes) State() State {
	return f.OutState
}

func (f *funcBlockRes) Output() anyvec.Vector {
	return f.OutRes.Output()
}

func (f *funcBlockRes) Vars() anydiff.VarSet {
	return f.V
}

func (f *funcBlockRes) Propagate(u anyvec.Vector, s StateGrad,
	g anydiff.Grad) (anyvec.Vector, StateGrad) {
	c := f.InPool.Vector.Creator()
	g[f.InPool] = c.MakeVector(f.InPool.Output().Len())
	g[f.StatePool] = c.MakeVector(f.StatePool.Output().Len())
	f.OutRes.Propagate(u, g)
	if s != nil {
		v := s.(*FuncBlockState).Vector
		f.StateRes.Propagate(v, g)
	}
	inGrad := g[f.InPool]
	stateGrad := g[f.StatePool]
	delete(g, f.InPool)
	delete(g, f.StatePool)
	return inGrad, &FuncBlockState{
		VecState: &VecState{
			Vector:     stateGrad,
			PresentMap: f.OutState.PresentMap,
		},
		StartRes: f.OutState.StartRes,
	}
}
