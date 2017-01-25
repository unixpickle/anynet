package anyconv

import (
	"errors"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyff"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
)

// A PostTrainer uses a list of samples to replace
// BatchNorm layers with hard-wired affine transforms.
type PostTrainer struct {
	Samples anysgd.SampleList

	// Fetcher must return *anyff.Batch instances.
	Fetcher anysgd.Fetcher

	// BatchSize specifies how many samples to feed to the
	// network at once.
	BatchSize int

	Net anynet.Net
}

// Run performs layer replacement.
//
// It returns with an error if the Fetcher failed.
// However, even in the event of an error, some layers may
// already be replaced with affine transforms.
func (p *PostTrainer) Run() error {
	for i, x := range p.Net {
		bn, ok := x.(*BatchNorm)
		if !ok {
			continue
		}
		preNet := p.Net[:i]
		mean, stddev, err := momentsFromOutputs(bn, p.evaluateBatch(preNet))
		if err != nil {
			return err
		}
		scaler := bn.Scalers.Vector.Copy()
		scaler.Div(stddev)
		bias := bn.Biases.Vector.Copy()
		mean.Mul(scaler)
		bias.Sub(mean)
		p.Net[i] = &anynet.Affine{
			Scalers: anydiff.NewVar(scaler),
			Biases:  anydiff.NewVar(bias),
		}
	}
	return nil
}

func (p *PostTrainer) evaluateBatch(subNet anynet.Net) <-chan *postTrainerOutput {
	resChan := make(chan *postTrainerOutput, 1)
	go func() {
		defer close(resChan)
		for i := 0; i < p.Samples.Len(); i += p.BatchSize {
			bs := p.Samples.Len() - i
			if bs > p.BatchSize {
				bs = p.BatchSize
			}
			batch, err := p.Fetcher.Fetch(p.Samples.Slice(i, i+bs))
			if err != nil {
				resChan <- &postTrainerOutput{Err: err}
				return
			}
			inVec := batch.(*anyff.Batch).Inputs
			outVec := subNet.Apply(inVec, bs).Output().Copy()
			resChan <- &postTrainerOutput{
				Vec: outVec,
			}
		}
	}()
	return resChan
}

type postTrainerOutput struct {
	Err error
	Vec anyvec.Vector
}

func momentsFromOutputs(b *BatchNorm, c <-chan *postTrainerOutput) (mean,
	stddev anyvec.Vector, err error) {
	var sum, sqSum anyvec.Vector
	var count int
	for item := range c {
		if item.Err != nil {
			return nil, nil, item.Err
		}

		count += item.Vec.Len() / b.InputCount
		thisSum := anyvec.SumRows(item.Vec, b.InputCount)
		item.Vec.Mul(item.Vec.Copy())
		thisSqSum := anyvec.SumRows(item.Vec, b.InputCount)

		if sum == nil {
			sum = thisSum
			sqSum = thisSqSum
		} else {
			sum.Add(thisSum)
			sqSum.Add(thisSqSum)
		}
	}
	if sum == nil {
		return nil, nil, errors.New("no samples to average")
	}
	normalizer := sum.Creator().MakeNumeric(1 / float64(count))
	sum.Scale(normalizer)
	sqSum.Scale(normalizer)

	sumSq := sum.Copy()
	sumSq.Mul(sum)

	sqSum.Sub(sumSq)
	sqSum.AddScaler(sqSum.Creator().MakeNumeric(b.stabilizer()))
	anyvec.Pow(sqSum, sqSum.Creator().MakeNumeric(0.5))

	return sum, sqSum, nil
}
