package anyrnn

import (
	"reflect"
	"testing"

	"github.com/unixpickle/anyvec/anyvec32"
)

func TestVecStateReduce(t *testing.T) {
	s := &VecState{
		Vector: anyvec32.MakeVectorData([]float32{
			1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
		}),
		PresentMap: []bool{true, false, true, true, false, false, true, true},
	}
	reduced := s.Reduce([]bool{true, false, false, true, false, false, false, true})
	expected := []float32{1, 2, 5, 6, 9, 10}
	actual := reduced.(*VecState).Vector.Data().([]float32)
	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("expected %v but got %v", expected, actual)
	}

	reduced = s.Reduce([]bool{false, false, true, false, false, false, true, false})
	expected = []float32{3, 4, 7, 8}
	actual = reduced.(*VecState).Vector.Data().([]float32)
	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("expected %v but got %v", expected, actual)
	}
}

func TestVecStateExpand(t *testing.T) {
	s := &VecState{
		Vector: anyvec32.MakeVectorData([]float32{
			1, 2, 3, 4, 5, 6,
		}),
		PresentMap: []bool{true, false, true, false, false, false, false, true},
	}
	expanded := s.Expand([]bool{true, false, true, false, true, false, true, true})
	expected := []float32{1, 2, 3, 4, 0, 0, 0, 0, 5, 6}
	actual := expanded.(*VecState).Vector.Data().([]float32)
	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("expected %v but got %v", expected, actual)
	}

	expanded = s.Expand([]bool{true, true, true, true, true, true, true, true})
	expected = []float32{1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 5, 6}
	actual = expanded.(*VecState).Vector.Data().([]float32)
	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("expected %v but got %v", expected, actual)
	}

	expanded = s.Expand([]bool{true, false, true, false, true, true, true, true})
	expected = []float32{1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 5, 6}
	actual = expanded.(*VecState).Vector.Data().([]float32)
	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("expected %v but got %v", expected, actual)
	}
}
