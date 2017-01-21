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

func TestLayerBlock(t *testing.T) {
	inVars := []*anydiff.Var{}
	inBatches := []*anyseq.ResBatch{}
	for i := 0; i < 3; i++ {
		vec := anyvec32.MakeVector(9)
		anyvec.Rand(vec, anyvec.Normal, nil)
		v := anydiff.NewVar(vec)
		batch := &anyseq.ResBatch{
			Packed:  v,
			Present: []bool{true, true, true},
		}
		inVars = append(inVars, v)
		inBatches = append(inBatches, batch)
	}
	inSeq := anyseq.ResSeq(inBatches)

	block := &LayerBlock{
		Layer: anynet.Net{
			anynet.NewFC(anyvec32.CurrentCreator(), 3, 2),
			anynet.Tanh,
		},
	}

	if len(block.Parameters()) != 2 {
		t.Errorf("expected 2 parameters, but got %d", len(block.Parameters()))
	}
	checker := &anydifftest.SeqChecker{
		F: func() anyseq.Seq {
			return Map(inSeq, block)
		},
		V: append(inVars, block.Parameters()...),
	}
	checker.FullCheck(t)
}
