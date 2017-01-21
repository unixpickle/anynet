package anyrnn

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
)

// A Stack is a meta-Block for composing Blocks.
// In a Stack, the first Block's output is fed as input to
// the next Block, etc.
//
// An empty Stack is invalid.
type Stack []Block

// Start produces a start state.
func (s Stack) Start(n int) State {
	s.assertNonEmpty()
	res := make(stackState, len(s))
	for i, x := range s {
		res[i] = x.Start(n)
	}
	return res
}

// PropagateStart back-propagates through the start state.
func (s Stack) PropagateStart(sg StateGrad, g anydiff.Grad) {
	for i, x := range s {
		x.PropagateStart(sg.(stackGrad)[i], g)
	}
}

// Step applies the block for a single timestep.
func (s Stack) Step(st State, in anyvec.Vector) Res {
	res := &stackRes{V: anydiff.VarSet{}}
	inVec := in
	for i, x := range s {
		inState := st.(stackState)[i]
		blockRes := x.Step(inState, inVec)
		inVec = blockRes.Output()
		res.Reses = append(res.Reses, blockRes)
		res.OutState = append(res.OutState, blockRes.State())
		res.V = anydiff.MergeVarSets(res.V, blockRes.Vars())
	}
	return res
}

func (s Stack) assertNonEmpty() {
	if len(s) == 0 {
		panic("empty Stack is invalid")
	}
}

type stackRes struct {
	Reses    []Res
	OutState stackState
	V        anydiff.VarSet
}

func (s *stackRes) State() State {
	return s.OutState
}

func (s *stackRes) Output() anyvec.Vector {
	return s.Reses[len(s.Reses)-1].Output()
}

func (s *stackRes) Vars() anydiff.VarSet {
	return s.V
}

func (s *stackRes) Propagate(u anyvec.Vector, sg StateGrad, g anydiff.Grad) (anyvec.Vector,
	StateGrad) {
	downVec := u
	downStates := make(stackGrad, len(s.Reses))
	for i := len(s.Reses); i >= 0; i-- {
		var stateUpstream StateGrad
		if sg != nil {
			stateUpstream = sg.(stackGrad)[i]
		}
		down, downState := s.Reses[i].Propagate(downVec, stateUpstream, g)
		downVec = down
		downStates[i] = downState
	}
	return downVec, downStates
}

type stackState []State

func (s stackState) Present() PresentMap {
	return s[0].Present()
}

func (s stackState) Reduce(p PresentMap) State {
	res := make(stackState, len(s))
	for i, x := range s {
		res[i] = x.Reduce(p)
	}
	return res
}

type stackGrad []StateGrad

func (s stackGrad) Present() PresentMap {
	return s[0].Present()
}

func (s stackGrad) Expand(p PresentMap) StateGrad {
	res := make(stackGrad, len(s))
	for i, x := range s {
		res[i] = x.Expand(p)
	}
	return res
}
