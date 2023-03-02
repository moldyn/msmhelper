MACROTRAJ="../../test/assets/8state_macrotraj"
MICROTRAJ="../../test/assets/8state_microtraj"

## WAITING TIME DISTRIBUTION
WTD_ARGS="--start 1 --final 4 --nsteps 10000000 --max-lagtime 25 --frames-per-unit 1 --unit frames" 
python -m msmhelper waiting-time-dist $WTD_ARGS \
    --filename $MACROTRAJ -o ${MACROTRAJ}.wtd.jpg
python -m msmhelper waiting-time-dist $WTD_ARGS \
    --filename $MACROTRAJ --microfilename $MICROTRAJ -o ${MACROTRAJ}.sh.wtd.jpg

## WAITING TIME DISTRIBUTION
WTS_ARGS="--lagtimes 1 2 3 --start 1 --final 4 --nsteps 10000000 --frames-per-unit 1 --unit frames" 
python -m msmhelper waiting-times $WTS_ARGS \
    --filename $MACROTRAJ -o ${MACROTRAJ}.wts.jpg
python -m msmhelper waiting-times $WTS_ARGS \
    --filename $MACROTRAJ --microfilename $MICROTRAJ -o ${MACROTRAJ}.sh.wts.jpg

## IMPLIED TIMESCALES
IMPL_ARGS="--max-lagtime 25 --frames-per-unit 1 --unit frames" 
python -m msmhelper implied-timescales $IMPL_ARGS \
    --filename $MACROTRAJ -o ${MACROTRAJ}.impl.jpg
python -m msmhelper implied-timescales $IMPL_ARGS \
    --filename $MACROTRAJ --microfilename $MICROTRAJ -o ${MACROTRAJ}.sh.impl.jpg

## CKTEST
CKTEST_ARGS="--lagtimes 1 2 3 4 5 --max-time 500 --frames-per-unit 1 --unit frames --grid 2 2" 
python -m msmhelper ck-test $CKTEST_ARGS \
    --filename $MACROTRAJ -o ${MACROTRAJ}.cktest.jpg
python -m msmhelper ck-test $CKTEST_ARGS \
    --filename $MACROTRAJ --microfilename $MICROTRAJ -o ${MACROTRAJ}.sh.cktest.jpg
