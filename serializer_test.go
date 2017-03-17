package anynet

import (
	"reflect"
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/serializer"
)

func TestActivationSerialize(t *testing.T) {
	a1 := Tanh
	a2 := LogSoftmax
	a3 := Sigmoid
	a4 := ReLU
	a5 := Sin
	a6 := Exp
	data, err := serializer.SerializeAny(a1, a2, a3, a4, a5, a6)
	if err != nil {
		t.Fatal(err)
	}
	var newA1, newA2, newA3, newA4, newA5, newA6 Activation
	err = serializer.DeserializeAny(data, &newA1, &newA2, &newA3, &newA4, &newA5, &newA6)
	if err != nil {
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
	if newA4 != a4 {
		t.Error("ReLU failed")
	}
	if newA5 != a5 {
		t.Error("Sin failed")
	}
	if newA6 != a6 {
		t.Error("Exp failed")
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
	if !reflect.DeepEqual(fc, newFC) {
		t.Fatal("incorrect result")
	}
}

func TestAffineSerialize(t *testing.T) {
	affine := &Affine{
		Scalers: anydiff.NewVar(anyvec32.MakeVectorData([]float32{1, 2, -1})),
		Biases:  anydiff.NewVar(anyvec32.MakeVectorData([]float32{-3, 1})),
	}
	data, err := serializer.SerializeAny(affine)
	if err != nil {
		t.Fatal(err)
	}
	var newAffine *Affine
	if err := serializer.DeserializeAny(data, &newAffine); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(affine, newAffine) {
		t.Fatal("incorrect result")
	}
}

func TestDropoutSerialize(t *testing.T) {
	do := &Dropout{Enabled: true, KeepProb: 0.335}
	data, err := serializer.SerializeAny(do)
	if err != nil {
		t.Fatal(err)
	}
	var do1 *Dropout
	if err := serializer.DeserializeAny(data, &do1); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(do, do1) {
		t.Fatal("incorrect result")
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

func TestAddMixerSerializer(t *testing.T) {
	c := anyvec32.DefaultCreator{}
	a := &AddMixer{
		In1: NewFC(c, 5, 3),
		In2: NewFC(c, 2, 3),
		Out: NewFC(c, 3, 1),
	}
	data, err := serializer.SerializeAny(a)
	if err != nil {
		t.Fatal(err)
	}
	var a1 *AddMixer
	if err := serializer.DeserializeAny(data, &a1); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a1, a) {
		t.Error("incorrect result")
	}
}

func TestConcatMixerSerializer(t *testing.T) {
	mixer := ConcatMixer{}
	data, err := serializer.SerializeAny(mixer)
	if err != nil {
		t.Fatal(err)
	}
	var mixer1 ConcatMixer
	if err := serializer.DeserializeAny(data, &mixer1); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(mixer1, mixer) {
		t.Error("incorrect result")
	}
}

func TestParamHiderSerialize(t *testing.T) {
	net := &ParamHider{Layer: Tanh}
	data, err := serializer.SerializeAny(net)
	if err != nil {
		t.Fatal(err)
	}
	var net1 *ParamHider
	if err := serializer.DeserializeAny(data, &net1); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(net, net1) {
		t.Fatal("incorrect result")
	}
}
