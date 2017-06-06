package anysgd

import (
	"errors"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec/anyvecsave"
	"github.com/unixpickle/serializer"
)

var errVarsGradMismatch = errors.New("variable list does not match gradients")

func marshalGradient(vars []*anydiff.Var, grad anydiff.Grad) ([]byte, error) {
	if grad == nil {
		return []byte{}, nil
	}
	if len(vars) != len(grad) {
		return nil, errVarsGradMismatch
	}

	var vecObjs []interface{}
	for _, v := range vars {
		vec, ok := grad[v]
		if !ok {
			return nil, errVarsGradMismatch
		}
		vecObjs = append(vecObjs, &anyvecsave.S{Vector: vec})
	}

	return serializer.SerializeAny(vecObjs...)
}

func unmarshalGradient(vars []*anydiff.Var, data []byte) (anydiff.Grad, error) {
	if len(data) == 0 {
		return nil, nil
	}

	var dests []interface{}
	for _ = range vars {
		dests = append(dests, new(*anyvecsave.S))
	}
	if err := serializer.DeserializeAny(data, dests...); err != nil {
		return nil, err
	}

	res := anydiff.Grad{}
	for i, v := range vars {
		vec := (*dests[i].(**anyvecsave.S)).Vector
		if vec.Len() != v.Vector.Len() {
			return nil, errors.New("bad vector length")
		} else if vec.Creator() != v.Vector.Creator() {
			return nil, errors.New("bad vector creator")
		}
		res[v] = vec
	}

	return res, nil
}
