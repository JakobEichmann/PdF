public class E04_SubConst_OK {
    @Interval(min = 1, max = 3)
    int f;

    public void foo() {
        f = 2;
        @Interval(min = 0, max = 2)
        int l = f - 1; // EXPECT: OK
    }
}
