public class E05_SubConst_AnnotationMismatch_FAIL {
    @Interval(min = 1, max = 3)
    int f;

    public void foo() {
        f = 2;
        // f - 1 in [0,2], but annotation says [2,4]
        @Interval(min = 2, max = 4)
        int l = f - 1; // EXPECT: FAIL
    }
}
