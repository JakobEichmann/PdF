public class E12_DivSafe_OK {
    @Interval(min = 1, max = 2)
    int g;

    public void foo() {
        g = 2;
        // 10 / g in [5,10] if g in [1,2] and integers (over-approx ok)
        @Interval(min = 5, max = 10)
        int l = 10 / g; // EXPECT: OK (needs conservative rule)
    }
}
