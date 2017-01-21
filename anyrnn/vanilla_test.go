package anyrnn

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anydifftest"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
)

func TestVanillaOutput(t *testing.T) {
	v := &Vanilla{
		InCount:  2,
		OutCount: 3,
		StateWeights: anydiff.NewVar(anyvec32.MakeVectorData([]float32{
			1.013949080929492, 0.651993107300643, 1.962063017373509,
			-0.305518636912932, -1.907428571394675, 1.047494354506540,
			0.424928126971939, 1.152028175999884, -0.159508838475856,
		})),
		InputWeights: anydiff.NewVar(anyvec32.MakeVectorData([]float32{
			0.578969569547752, -1.738131402776219,
			1.834645967361668, 0.216295204977240,
			-0.174414967388466, 0.495420882674173,
		})),
		Biases: anydiff.NewVar(anyvec32.MakeVectorData([]float32{
			-1.787774931407171, 0.295051579270469, 0.926511486922573,
		})),
		StartState: anydiff.NewVar(anyvec32.MakeVectorData([]float32{
			1.137478454905680, 0.318992795539938, -0.263672034086737,
		})),
		Activation: anynet.Tanh,
	}
	seq := anyseq.ConstSeq([]*anyseq.Batch{
		{
			Packed:  anyvec32.MakeVectorData([]float32{1, 2, -1, -3}),
			Present: []bool{true, true, false},
		},
		{
			Packed:  anyvec32.MakeVectorData([]float32{2, -1}),
			Present: []bool{false, true, false},
		},
		{
			Packed:  anyvec32.MakeVectorData([]float32{-1.5, 2}),
			Present: []bool{false, true, false},
		},
	})
	actual := Map(seq, v).Output()
	expected := []*anyseq.Batch{
		{
			Packed: anyvec32.MakeVectorData([]float32{
				-0.999078474068952, 0.869277718016662, 0.989782342296173,
				0.998757641871079, -0.997864863237772, 0.468039628893154,
			}),
			Present: []bool{true, true, false},
		},
		{
			Packed: anyvec32.MakeVectorData([]float32{
				0.983305063303649, 0.999982959700321, -0.615398128318720,
			}),
			Present: []bool{false, true, false},
		},
		{
			Packed: anyvec32.MakeVectorData([]float32{
				-0.999977199810133, -0.999883828747684, 0.999089273224046,
			}),
			Present: []bool{false, true, false},
		},
	}
	if !seqsEquivalent(actual, expected) {
		t.Errorf("expected %s but got %s", seqString(expected), seqString(actual))
	}
}

func TestVanillaProp(t *testing.T) {
	block := NewVanilla(anyvec32.CurrentCreator(), 2, 3, anynet.Tanh)
	inVars := []*anydiff.Var{
		anydiff.NewVar(anyvec32.MakeVectorData([]float32{
			1.000009682963246, 0.887353762043918,
			1.390648176281434, -0.709610839726816,
		})),
		anydiff.NewVar(anyvec32.MakeVectorData([]float32{
			1.46841279925354, -1.6971931951273,
		})),
		anydiff.NewVar(anyvec32.MakeVectorData([]float32{
			-1.567780854880226, 0.639114679829077,
		})),
	}
	seq := anyseq.ResSeq([]*anyseq.ResBatch{
		{
			Packed:  inVars[0],
			Present: []bool{true, true, false},
		},
		{
			Packed:  inVars[1],
			Present: []bool{false, true, false},
		},
		{
			Packed:  inVars[2],
			Present: []bool{false, true, false},
		},
	})
	if len(block.Parameters()) != 4 {
		t.Errorf("expected 4 parameters, but got %d", len(block.Parameters()))
	}
	checker := &anydifftest.SeqChecker{
		F: func() anyseq.Seq {
			return Map(seq, block)
		},
		V: append(inVars, block.Parameters()...),
	}
	checker.FullCheck(t)
}

func seqString(s []*anyseq.Batch) string {
	var parts []string
	for _, x := range s {
		parts = append(parts, fmt.Sprintf("{Packed: %v, Present: %v}", x.Packed.Data(),
			x.Present))
	}
	return "[" + strings.Join(parts, " ") + "]"
}

func seqsEquivalent(s1, s2 []*anyseq.Batch) bool {
	if len(s1) != len(s2) {
		return false
	}
	for i, b1 := range s1 {
		b2 := s2[i]
		if !reflect.DeepEqual(b1.Present, b2.Present) {
			return false
		}
		diff := b1.Packed.Copy()
		diff.Sub(b2.Packed)
		max := anyvec.AbsMax(diff)
		switch max := max.(type) {
		case float32:
			if max > 1e-3 {
				return false
			}
		case float64:
			if max > 1e-5 {
				return false
			}
		default:
			panic(fmt.Sprintf("unsupported numeric type: %T", max))
		}
	}
	return true
}
