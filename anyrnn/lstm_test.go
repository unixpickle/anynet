package anyrnn

import (
	"testing"

	"github.com/unixpickle/anydiff/anydifftest"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec/anyvec32"
)

func TestLSTMProp(t *testing.T) {
	inSeq, inVars := randomTestSequence(anyvec32.CurrentCreator(), 3)
	block := NewLSTM(anyvec32.CurrentCreator(), 3, 2)
	if len(block.Parameters()) != 18 {
		t.Errorf("expected 18 parameters, but got %d", len(block.Parameters()))
	}
	checker := &anydifftest.SeqChecker{
		F: func() anyseq.Seq {
			return Map(inSeq, block)
		},
		V: append(inVars, block.Parameters()...),
	}
	checker.FullCheck(t)
}
