package anynet

import (
	"reflect"
	"testing"

	"github.com/unixpickle/serializer"
)

func TestActivationSerialize(t *testing.T) {
	a1 := Tanh
	a2 := LogSoftmax
	data, err := serializer.SerializeAny(a1, a2)
	if err != nil {
		t.Fatal(err)
	}
	var newA1, newA2 Activation
	if err := serializer.DeserializeAny(data, &newA1, &newA2); err != nil {
		t.Fatal(err)
	}
	if newA1 != a1 {
		t.Error("Tanh failed")
	}
	if newA2 != a2 {
		t.Error("LogSoftmax failed")
	}
}

func TestNetSerialize(t *testing.T) {
	net := Net{Tanh, LogSoftmax}
	data, err := serializer.SerializeAny(net)
	if err != nil {
		t.Fatal(err)
	}
	var net1 Net
	if err := serializer.DeserializeAny(data, &net1); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(net, net1) {
		t.Fatal("networks not equal")
	}
}
