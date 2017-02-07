package anyrnn

import (
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anydifftest"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
)

func TestFuncBlock(t *testing.T) {
	c := anyvec32.CurrentCreator()
	inSeq, inVars := randomTestSequence(3)
	stateFC := anynet.NewFC(c, 2, 2)
	inputFC := anynet.NewFC(c, 3, 2)
	outFC := anynet.NewFC(c, 2, 3)
	startState := anydiff.NewVar(c.MakeVector(2))
	anyvec.Rand(startState.Vector, anyvec.Normal, nil)

	params := append(append(append(append(inVars, stateFC.Parameters()...),
		inputFC.Parameters()...), outFC.Parameters()...), startState)

	block := &FuncBlock{
		Func: func(in, state anydiff.Res, n int) (out, newState anydiff.Res) {
			st := stateFC.Apply(state, n)
			it := inputFC.Apply(in, n)
			newState = anydiff.Add(st, it)
			out = outFC.Apply(newState, n)
			return
		},
		MakeStart: func(n int) anydiff.Res {
			zeroVec := anydiff.NewConst(c.MakeVector(n * startState.Vector.Len()))
			return anydiff.AddRepeated(zeroVec, startState)
		},
	}

	checker := &anydifftest.SeqChecker{
		F: func() anyseq.Seq {
			return Map(inSeq, block)
		},
		V: params,
	}
	checker.FullCheck(t)
}

func TestFuncBlockNilOut(t *testing.T) {
	c := anyvec32.CurrentCreator()
	inSeq, inVars := randomTestSequence(3)
	fcBlock := anynet.NewFC(c, 3, 3)
	startState := anydiff.NewVar(c.MakeVector(3))
	anyvec.Rand(startState.Vector, anyvec.Normal, nil)

	params := append(append(inVars, fcBlock.Parameters()...), startState)

	block := &FuncBlock{
		Func: func(in, state anydiff.Res, n int) (out, newState anydiff.Res) {
			return nil, anynet.Tanh.Apply(anydiff.Add(
				fcBlock.Apply(in, n),
				fcBlock.Apply(state, n),
			), n)
		},
		MakeStart: func(n int) anydiff.Res {
			zeroVec := anydiff.NewConst(c.MakeVector(n * startState.Vector.Len()))
			return anydiff.AddRepeated(zeroVec, startState)
		},
	}

	checker := &anydifftest.SeqChecker{
		F: func() anyseq.Seq {
			return Map(inSeq, block)
		},
		V: params,
	}
	checker.FullCheck(t)
}
