import unittest


# python -m unittest -v test_pipeline.py

# test_sample.py 내용들 옮겨보자

# 1) assertEqual(A, B, Msg) - A, B가 같은지 테스트
# 2) assertNotEqual(A, B, Msg) - A, B가 다른지 테스트
# 3) assertTrue(A, Msg) - A가 True인지 테스트
# 4) assertFalse(A, Msg) - A가 False인지 테스트
# 5) assertIs(A, B, Msg) - A, B가 동일한 객체인지 테스트
# 6) assertIsNot(A, B, Msg) - A, B가 동일하지 않는 객체인지 테스트
# 7) assertIsNone(A, Msg) - A가 None인지 테스트
# 8) assertIsNotNone(A, Msg) - A가 Not None인지 테스트
# 9) assertRaises(ZeroDivisionError, myCalc.add, 4, 0) - 특정 에러 확인


# setup (setUp) > running > finish (teardown)
class MyTests(unittest.TestCase):
    result = 0

    # initialize 같은 setup 필요할걸?
    def setUp(self):
        # 테스트 시작 전 셋업한다. 각 test마다 매번 호출된다
        # 매 테스트 메소드 실행 전 동작
        # self.cal = Calculator()
        self.aa = 10
        self.bb = 20

    def tearDown(self):
        del self.aa
        del self.bb

    def test_one_plus_two(self):
        self.assertEqual(1 + 4, 5)

    def test_other_assertions(self):
        self.assertTrue(1 == 1)
        self.assertFalse(1 == 2)
        self.assertGreater(2, 1)
        self.assertLess(1, 2)
        self.assertIn(1, [1, 2])
        self.assertIsInstance(1, int)

    def test_exceptions(self):
        with self.assertRaises(ZeroDivisionError):
            1 / 0
        with self.assertRaises(TypeError):
            1 + '2'


if __name__ == '__main__':
    unittest.main()
