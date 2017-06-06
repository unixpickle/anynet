package anysgd

import (
	"math/rand"
	"reflect"
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec64"
)

func TestGradientMarshal(t *testing.T) {
	c := anyvec64.DefaultCreator{}
	vars := randomVars(c)
	grad := randomGrad(vars)

	data, err := marshalGradient(vars, grad)
	if err != nil {
		t.Fatal(err)
	}

	newGrad, err := unmarshalGradient(vars, data)
	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(grad, newGrad) {
		t.Error("gradient mismatch")
	}
}

func testMarshal(t *testing.T, inst TransformMarshaler, v []*anydiff.Var) {
	var inGrads []anydiff.Grad
	var outGrads []anydiff.Grad
	var checkpoints [][]byte

	for i := 0; i < 5; i++ {
		inGrad := randomGrad(v)
		data, err := inst.MarshalBinary()
		if err != nil {
			t.Fatal(err)
		}
		outGrad := copyGrad(inst.Transform(copyGrad(inGrad)))
		inGrads = append(inGrads, inGrad)
		checkpoints = append(checkpoints, data)
		outGrads = append(outGrads, outGrad)
	}

	for _, i := range []int{2, 0, 3, 4, 1} {
		err := inst.UnmarshalBinary(checkpoints[i])
		if err != nil {
			t.Fatal(err)
		}
		in := inGrads[i]
		out := inst.Transform(in)
		if !reflect.DeepEqual(out, outGrads[i]) {
			t.Errorf("gradient %d came out wrong", i)
		}
	}
}

func randomVars(c anyvec.Creator) []*anydiff.Var {
	vars := []*anydiff.Var{}
	for i := 0; i < 20; i++ {
		size := i * rand.Intn(3)
		vec := c.MakeVector(size)
		anyvec.Rand(vec, anyvec.Normal, nil)
		v := anydiff.NewVar(vec)
		vars = append(vars, v)
	}
	return vars
}

func randomGrad(vars []*anydiff.Var) anydiff.Grad {
	resGrad := anydiff.Grad{}
	for _, v := range vars {
		gradVec := v.Vector.Creator().MakeVector(v.Vector.Len())
		anyvec.Rand(gradVec, anyvec.Normal, nil)
		resGrad[v] = gradVec
	}
	return resGrad
}
