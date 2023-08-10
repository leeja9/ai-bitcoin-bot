import unittest
# import random
# update this import statement for what you're testing
# from somefile.py import method as testmethodName


class TestCase(unittest.TestCase):
    # example test
    def test(self):
        pass
    #     test_val = None
    #     expected = ''
    #     test_case = testmethod(test_val)
    #     self.assertEqual(test_case, expected)

# generic code for generating random tests

# for generating random tests and seeing individual test results
# def _build_test(test_case, expected):
#     def test(self):
#         self.assertEqual(test_case, expected)
#     return test


# def generate_random_tests(n_tests, min_int, max_int):
#     rand_nums = set()
#     for _ in range(n_tests):
#         rand_nums.add(random.randint(min_int, max_int))
#     for rand_num in rand_nums:
#         # for random T/F
#         # rand_bool = random.choice((True, False))

#         test_case = testmethod(rand_num)
#         expected = ''
#         new_test = _build_test(test_case, expected)
#         test_name = 'test_rand_num_{}'.format(rand_num)
#         setattr(TestCase, test_name, new_test)


if __name__ == '__main__':
    # generate_random_tests(10, 1, 100)
    unittest.main()
