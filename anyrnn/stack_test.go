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

func TestStackOutput(t *testing.T) {
	layer1 := anynet.NewFC(anyvec32.CurrentCreator(), 3, 2)
	layer2 := anynet.Tanh

	input := anyvec32.MakeVectorData([]float32{
		2.098950, -0.645579, 2.106542,
		0.085620, 0.762207, -0.279375,
		0.993967, 2.453542, 1.729150,
		-0.971805, -0.315578, -0.306942,
	})
	inRes := anydiff.NewConst(input)
	expected := anynet.Net{layer1, layer2}.Apply(inRes, 4).Output()

	stacked := Stack{&LayerBlock{Layer: layer1}, &LayerBlock{Layer: layer2}}
	state := stacked.Start(4)
	actual := stacked.Step(state, input).Output()

	diff := actual.Copy()
	diff.Sub(expected)
	max := anyvec.AbsMax(diff).(float32)
	if max > 1e-3 {
		t.Errorf("expected %v but got %v", expected.Data(), actual.Data())
	}
}

func TestStackProp(t *testing.T) {
	inSeq, inVars := randomTestSequence(anyvec32.CurrentCreator(), 3)
	block := Stack{
		&LayerBlock{
			Layer: anynet.NewFC(anyvec32.CurrentCreator(), 3, 2),
		},
		&LayerBlock{
			Layer: anynet.Tanh,
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
