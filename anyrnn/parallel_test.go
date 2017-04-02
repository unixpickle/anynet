package anyrnn

import (
	"testing"

	"github.com/unixpickle/anydiff/anydifftest"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec/anyvec32"
)

func TestParallelProp(t *testing.T) {
	inSeq, inVars := randomTestSequence(anyvec32.CurrentCreator(), 3)
	block1 := NewLSTM(anyvec32.CurrentCreator(), 3, 1)
	block2 := NewLSTM(anyvec32.CurrentCreator(), 3, 2)
	block := &Parallel{
		Block1: block1,
		Block2: block2,
		Mixer:  anynet.ConcatMixer{},
	}
	if len(block.Parameters()) != 36 {
		t.Errorf("expected 36 parameters, but got %d", len(block.Parameters()))
	}
	checker := &anydifftest.SeqChecker{
		F: func() anyseq.Seq {
			return Map(inSeq, block)
		},
		V: append(inVars, block.Parameters()...),
	}
	checker.FullCheck(t)
}
