public class E03_AddConst_OK {
    @Interval(min = 1, max = 3)
    int f;

    public void foo() {
        f = 3;
        @Interval(min = 2, max = 4)
        int l = f + 1; // EXPECT: OK (needs shift rule)
    }
}
