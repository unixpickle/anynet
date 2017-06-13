package anymisc

import (
	"math"
	"reflect"
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec64"
	"github.com/unixpickle/serializer"
)

func TestSELU(t *testing.T) {
	c := anyvec64.DefaultCreator{}
	vec := c.MakeVector(512)
	anyvec.Rand(vec, anyvec.Normal, nil)

	expectedData := make([]float64, vec.Len())
	for i, in := range vec.Data().([]float64) {
		expectedData[i] = scalarSELU(in)
	}
	expected := c.MakeVectorData(expectedData)

	actual := (&SELU{}).Apply(anydiff.NewVar(vec), 4).Output()
	diff := actual.Copy()
	diff.Sub(expected)
	if anyvec.AbsMax(diff).(float64) > 1e-4 {
		t.Error("bad output vector")
	}
}

func TestSELUSerialize(t *testing.T) {
	s := &SELU{Alpha: 3, Lambda: 5}
	data, err := serializer.SerializeAny(s)
	if err != nil {
		t.Fatal(err)
	}

	var s1 *SELU
	if err := serializer.DeserializeAny(data, &s1); err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(s, s1) {
		t.Fatal("bad value")
	}
}

func scalarSELU(x float64) float64 {
	if x > 0 {
		return seluDefaultLambda * x
	} else {
		return seluDefaultLambda * (seluDefaultAlpha*math.Exp(x) - seluDefaultAlpha)
	}
}
