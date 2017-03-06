for imp in 0 1e-2 1e-3
do
    for ort in 0 1e-2 1e-3
    do
        for w in 1 3 10
        do
            for unit in 16 32
            do
                for vocab in 2 4 8 32
                do
                    if test ${w} = 10 -a ${vocab} = 32 ; then
                        continue
                    fi
                    if test ${w} = 10 -a ${unit} = 32 ; then
                        continue
                    fi
                    echo logs/log.ort${ort}imp${imp}w${w}t1b512iu256u${unit}d02v${vocab}.log &
                    python -u scripts/train_mnist.py --gpu=1 --epoch 200 --batchsize 512 --out data0306/test_ort${ort}imp${imp}w${w}t1b512iu256u${unit}d02v${vocab} -ort ${ort} -imp ${imp} --image-unit 256 --unit ${unit} --word ${w} --turn 1 --dropout 0.2 --vocab ${vocab} > logs/log.ort${ort}imp${imp}w${w}t1b512iu256u${unit}d02v${vocab}.log &
                done
            done
        done
        python -u scripts/train_mnist.py --gpu=1 --epoch 190 --batchsize 512 --out gomidir -ort 1e-3 -imp 1e-3 --image-unit 256 --unit 32 --word 5 --turn 1 --dropout 0.2 -v 32 > log.gomi
    done
done
