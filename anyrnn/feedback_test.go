package anyrnn

import (
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anydifftest"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec64"
)

func TestFeedbackProp(t *testing.T) {
	c := anyvec64.CurrentCreator()
	vec := c.MakeVector(2)
	anyvec.Rand(vec, anyvec.Normal, nil)
	block := &Feedback{
		Mixer: &anynet.AddMixer{
			In1: anynet.NewFC(c, 3, 3),
			In2: anynet.NewFC(c, 2, 3),
			Out: anynet.Tanh,
		},
		Block:   NewLSTM(c, 3, 2),
		InitOut: anydiff.NewVar(vec),
	}
	inSeq, inVars := randomTestSequence(c, 3)
	if len(block.Parameters()) != 23 {
		t.Errorf("expected 23 parameters, but got %d", len(block.Parameters()))
	}
	checker := &anydifftest.SeqChecker{
		F: func() anyseq.Seq {
			return Map(inSeq, block)
		},
		V: append(inVars, block.Parameters()...),
	}
	checker.FullCheck(t)
}
