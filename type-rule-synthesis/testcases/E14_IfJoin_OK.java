public class E14_IfJoin_OK {
    @Interval(min = 1, max = 3)
    int f;

    public void foo(boolean b) {
        if (b) {
            f = 1;
        } else {
            f = 3;
        }
        @Interval(min = 2, max = 4)
        int l = f + 1; // EXPECT: OK (needs join + shift)
    }
}
