package anyrnn

import (
	"reflect"
	"testing"

	"github.com/unixpickle/anydiff/anydifftest"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec64"
	"github.com/unixpickle/serializer"
)

func TestMarkovOutput(t *testing.T) {
	for _, mode := range []string{"Concat", "DepthWise"} {
		t.Run(mode, func(t *testing.T) {
			c := anyvec64.DefaultCreator{}
			markov := NewMarkov(c, 2, 3, mode == "DepthWise")
			markov.StartState.Vector.SetData(
				c.MakeNumericList([]float64{-1, -2, -3, -4, -5, -6}),
			)

			inSeq := anyseq.ConstSeqList(c, [][]anyvec.Vector{
				{
					c.MakeVectorData(c.MakeNumericList([]float64{1, 2, 3})),
					c.MakeVectorData(c.MakeNumericList([]float64{4, 5, 6})),
					c.MakeVectorData(c.MakeNumericList([]float64{7, 8, 9})),
				},
			})

			actual := anyseq.SeparateSeqs(Map(inSeq, markov).Output())
			var expected [][]anyvec.Vector
			if mode == "Concat" {
				expected = [][]anyvec.Vector{
					{
						c.MakeVectorData(c.MakeNumericList([]float64{1, 2, 3, -1, -2, -3, -4, -5, -6})),
						c.MakeVectorData(c.MakeNumericList([]float64{4, 5, 6, 1, 2, 3, -1, -2, -3})),
						c.MakeVectorData(c.MakeNumericList([]float64{7, 8, 9, 4, 5, 6, 1, 2, 3})),
					},
				}
			} else {
				expected = [][]anyvec.Vector{
					{
						c.MakeVectorData(c.MakeNumericList([]float64{1, -1, -4, 2, -2, -5, 3, -3, -6})),
						c.MakeVectorData(c.MakeNumericList([]float64{4, 1, -1, 5, 2, -2, 6, 3, -3})),
						c.MakeVectorData(c.MakeNumericList([]float64{7, 4, 1, 8, 5, 2, 9, 6, 3})),
					},
				}
			}
			if !reflect.DeepEqual(actual, expected) {
				t.Errorf("expected %#v but got %#v", expected, actual)
			}
		})
	}
}

func TestMarkovGradients(t *testing.T) {
	c := anyvec64.DefaultCreator{}
	markov := NewMarkov(c, 2, 3, true)
	markov.StartState.Vector.SetData(
		c.MakeNumericList([]float64{-1, -2, -3, -4, -5, -6}),
	)
	inSeq, inVars := randomTestSequence(c, 3)
	checker := &anydifftest.SeqChecker{
		F: func() anyseq.Seq {
			return Map(inSeq, markov)
		},
		V: append(inVars, markov.Parameters()...),
	}
	checker.FullCheck(t)
}

func TestMarkovSerialize(t *testing.T) {
	c := anyvec64.DefaultCreator{}
	markov := NewMarkov(c, 2, 3, true)
	markov.StartState.Vector.SetData(
		c.MakeNumericList([]float64{-1, -2, -3, -4, -5, -6}),
	)
	data, err := serializer.SerializeAny(markov)
	if err != nil {
		t.Fatal(err)
	}
	var markov1 *Markov
	if err := serializer.DeserializeAny(data, &markov1); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(markov, markov1) {
		t.Error("bad deserialized value")
	}
}
