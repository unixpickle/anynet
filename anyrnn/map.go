package anyrnn

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
)

type mapRes struct {
	F        func(s StateGrad, g anydiff.Grad)
	InitPres PresentMap
	In       anyseq.Seq
	Out      []*anyseq.Batch
	BlockRes []Res
	Block    Block
	V        anydiff.VarSet
}

// Map maps a Block over an input sequence batch, giving
// an output sequence batch.
func Map(s anyseq.Seq, b Block) anyseq.Seq {
	inSteps := s.Output()
	if len(inSteps) == 0 {
		return &mapRes{}
	}

	state := b.Start(len(inSteps[0].Present))
	return MapWithStart(s, b, state, func(sg StateGrad, g anydiff.Grad) {
		b.PropagateStart(sg, g)
	})
}

// MapWithStart is like Map, but it takes a customized
// start state rather than using the block's default start
// state.
//
// During back-propagation, f is called with the upstream
// state gradient for the start state.
func MapWithStart(s anyseq.Seq, b Block, state State, f func(StateGrad, anydiff.Grad)) anyseq.Seq {
	inSteps := s.Output()
	if len(inSteps) == 0 {
		return &mapRes{}
	}

	initPres := state.Present()
	if inSteps[0].NumPresent() != len(inSteps[0].Present) {
		state = state.Reduce(inSteps[0].Present)
	}
	res := &mapRes{F: f, InitPres: initPres, In: s, Block: b, V: s.Vars()}

	for _, x := range inSteps {
		if x.NumPresent() != state.Present().NumPresent() {
			state = state.Reduce(x.Present)
		}
		step := b.Step(state, x.Packed)
		res.BlockRes = append(res.BlockRes, step)
		res.V = anydiff.MergeVarSets(res.V, step.Vars())
		res.Out = append(res.Out, &anyseq.Batch{
			Packed:  step.Output(),
			Present: x.Present,
		})
		state = step.State()
	}

	return res
}

func (m *mapRes) Output() []*anyseq.Batch {
	return m.Out
}

func (m *mapRes) Vars() anydiff.VarSet {
	return m.V
}

func (m *mapRes) Propagate(u []*anyseq.Batch, g anydiff.Grad) {
	if len(u) == 0 {
		return
	}

	var downstream []*anyseq.Batch
	if g.Intersects(m.In.Vars()) {
		downstream = make([]*anyseq.Batch, len(u))
	}

	var upState StateGrad
	for i := len(m.BlockRes) - 1; i >= 0; i-- {
		blockRes := m.BlockRes[i]
		if upState != nil {
			newPres := blockRes.State().Present()
			if newPres.NumPresent() != upState.Present().NumPresent() {
				upState = upState.Expand(newPres)
			}
		}
		down, downState := blockRes.Propagate(u[i].Packed, upState, g)
		if downstream != nil {
			downstream[i] = &anyseq.Batch{Packed: down, Present: u[i].Present}
		}
		upState = downState
	}

	if upState != nil {
		if m.InitPres.NumPresent() != upState.Present().NumPresent() {
			upState = upState.Expand(m.InitPres)
		}
		m.F(upState, g)
	}

	if downstream != nil {
		m.In.Propagate(downstream, g)
	}
}
