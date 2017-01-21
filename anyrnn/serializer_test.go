package anyrnn

import (
	"reflect"
	"testing"

	"github.com/unixpickle/anynet"
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
	v.Biases.Vector.AddScaler(float32(1))

	testSerialize(t, v)
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
