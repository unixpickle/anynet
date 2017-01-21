package anynet

import (
	"reflect"
	"testing"

	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/serializer"
)

func TestActivationSerialize(t *testing.T) {
	a1 := Tanh
	a2 := LogSoftmax
	a3 := Sigmoid
	data, err := serializer.SerializeAny(a1, a2, a3)
	if err != nil {
		t.Fatal(err)
	}
	var newA1, newA2, newA3 Activation
	if err := serializer.DeserializeAny(data, &newA1, &newA2, &newA3); err != nil {
		t.Fatal(err)
	}
	if newA1 != a1 {
		t.Error("Tanh failed")
	}
	if newA2 != a2 {
		t.Error("LogSoftmax failed")
	}
	if newA3 != a3 {
		t.Error("Sigmoid failed")
	}
}

func TestFCSerialize(t *testing.T) {
	fc := NewFC(anyvec32.DefaultCreator{}, 7, 5)
	data, err := serializer.SerializeAny(fc)
	if err != nil {
		t.Fatal(err)
	}
	var newFC *FC
	if err := serializer.DeserializeAny(data, &newFC); err != nil {
		t.Fatal(err)
	}
	if newFC.InCount != fc.InCount || newFC.OutCount != fc.OutCount {
		t.Fatalf("expected %d->%d but got %d->%d", fc.InCount, fc.OutCount,
			newFC.InCount, newFC.OutCount)
	}
	weightDiff := newFC.Weights.Vector.Copy()
	weightDiff.Sub(fc.Weights.Vector)
	if anyvec.AbsMax(weightDiff).(float32) > 1e-3 {
		t.Error("weight mismatch")
	}
	biasDiff := newFC.Biases.Vector.Copy()
	biasDiff.Sub(fc.Biases.Vector)
	if anyvec.AbsMax(biasDiff).(float32) > 1e-3 {
		t.Error("weight mismatch")
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
