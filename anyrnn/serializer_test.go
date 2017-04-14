package anyrnn

import (
	"reflect"
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/serializer"
)

func TestLayerSerialize(t *testing.T) {
	testSerialize(t, &LayerBlock{
		Layer: anynet.Tanh,
	})
}

func TestStackSerialize(t *testing.T) {
	testSerialize(t, Stack{
		&LayerBlock{Layer: anynet.Tanh},
		&LayerBlock{Layer: anynet.LogSoftmax},
	})
}

func TestVanillaSerialize(t *testing.T) {
	v := NewVanilla(anyvec32.CurrentCreator(), 3, 2, anynet.Tanh)

	// Make sure the biases are different than the init state.
	v.Biases.Vector.AddScalar(float32(1))

	testSerialize(t, v)
}

func TestLSTMGateSerialize(t *testing.T) {
	g := NewLSTMGate(anyvec32.CurrentCreator(), 3, 2, anynet.Sigmoid)

	// Make sure the biases are different than the init state.
	g.Biases.Vector.AddScalar(float32(1))

	testSerialize(t, g)
}

func TestLSTMSerialize(t *testing.T) {
	testSerialize(t, NewLSTM(anyvec32.CurrentCreator(), 3, 2))
}

func TestBidirSerialize(t *testing.T) {
	c := anyvec32.CurrentCreator()
	b := &Bidir{
		Forward:  NewVanilla(c, 5, 3, anynet.Tanh),
		Backward: NewVanilla(c, 5, 2, anynet.Tanh),
		Mixer: &anynet.AddMixer{
			In1: anynet.NewFC(c, 3, 2),
			In2: anynet.NewFC(c, 2, 2),
			Out: anynet.Tanh,
		},
	}
	testSerialize(t, b)
}

func TestFeedbackSerialize(t *testing.T) {
	c := anyvec32.CurrentCreator()
	vec := c.MakeVector(2)
	anyvec.Rand(vec, anyvec.Normal, nil)
	testSerialize(t, &Feedback{
		Mixer:   anynet.ConcatMixer{},
		Block:   NewLSTM(c, 3, 2),
		InitOut: anydiff.NewVar(vec),
	})
}

func TestParallelSerialize(t *testing.T) {
	c := anyvec32.CurrentCreator()
	b := &Parallel{
		Block1: NewVanilla(c, 5, 3, anynet.Tanh),
		Block2: NewVanilla(c, 5, 2, anynet.Tanh),
		Mixer: &anynet.AddMixer{
			In1: anynet.NewFC(c, 3, 2),
			In2: anynet.NewFC(c, 2, 2),
			Out: anynet.Tanh,
		},
	}
	testSerialize(t, b)
}

func testSerialize(t *testing.T, obj serializer.Serializer) {
	data, err := serializer.SerializeWithType(obj)
	if err != nil {
		t.Fatal(err)
	}
	newObj, err := serializer.DeserializeWithType(data)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(obj, newObj) {
		t.Errorf("expected %v but got %v", obj, newObj)
	}
}
