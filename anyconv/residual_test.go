package anyconv

import (
	"reflect"
	"testing"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/serializer"
)

func TestResidualSerialize(t *testing.T) {
	c := anyvec32.CurrentCreator()
	r := &Residual{
		Layer:      anynet.NewFC(c, 3, 2),
		Projection: anynet.NewFC(c, 3, 2),
	}
	data, err := serializer.SerializeAny(r)
	if err != nil {
		t.Fatal(err)
	}
	var newLayer *Residual
	if err = serializer.DeserializeAny(data, &newLayer); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(newLayer, r) {
		t.Fatal("layers differ")
	}

	r = &Residual{
		Layer: anynet.NewFC(c, 3, 2),
	}
	data, err = serializer.SerializeAny(r)
	if err != nil {
		t.Fatal(err)
	}
	if err = serializer.DeserializeAny(data, &newLayer); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(newLayer, r) {
		t.Fatal("layers differ")
	}
}
