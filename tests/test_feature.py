import unittest
import numpy as np
from lstm_tools.feature import Feature, Features

class TestFeature(unittest.TestCase):
    def setUp(self):
        # Setup test data
        self.value = 42.0
        self.name = "test_feature"
        self.feature = Feature(self.value, self.name)
        
    def test_creation(self):
        # Test basic creation
        self.assertEqual(float(self.feature), self.value)
        self.assertEqual(self.feature.name, self.name)
        self.assertEqual(self.feature._base.dtype, np.float32)  # Test default dtype
        
    def test_repr(self):
        # Test string representation
        self.assertEqual(repr(self.feature), f'Feature({self.name}: {self.value})')
        
    def test_add_operation(self):
        # Test adding an operation
        def test_op(x):
            return x * 2
            
        self.feature + test_op
        self.assertIn(test_op, self.feature.operations)
        
    def test_add_value(self):
        # Test adding a value
        result = self.feature + 10
        self.assertEqual(result, self.value + 10)

class TestFeatures(unittest.TestCase):
    def setUp(self):
        # Setup test data
        self.values = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.name = "test_features"
        self.features = [Feature(v, f"feature_{i}") for i, v in enumerate(self.values)]
        self.features_obj = Features(self.features, name=self.name)
        
    def test_creation(self):
        # Test basic creation
        self.assertEqual(len(self.features_obj), len(self.values))
        self.assertEqual(self.features_obj[0], self.values[0])
        
        # Test creation with invalid inputs
        with self.assertRaises(ValueError):
            Features(None)
        with self.assertRaises(ValueError):
            Features([])
        with self.assertRaises(TypeError):
            Features(42)  # Not a list or array
        
    def test_compressors(self):
        # Test adding compressors
        def test_comp(x):
            return np.mean(x) * 2
        test_comp.__name__ = "test_comp"
        
        self.features_obj.add_compressor(test_comp)
        self.assertIn(test_comp, self.features_obj.compressors)
        
        # Test adding invalid compressor
        with self.assertRaises(TypeError):
            self.features_obj.add_compressor("not_callable")
        
    def test_add_compressor(self):
        # Test adding compressor via + operator
        def custom_op(x): 
            return x.sum()
        custom_op.__name__ = "custom_sum"
        
        self.features_obj + custom_op
        self.assertIn(custom_op, self.features_obj.compressors)
        
        # Test adding via tuple
        self.features_obj + ("my_op", lambda x: x.mean())
        self.assertTrue(any(c.__name__ == "my_op" for c in self.features_obj.compressors))
        
        # Test invalid tuple format
        with self.assertRaises(ValueError):
            self.features_obj + (42, 42)
        
    def test_statistical_properties(self):
        # Test all statistical properties
        self.assertEqual(self.features_obj.mean, np.mean(self.values))
        self.assertEqual(self.features_obj.min, np.min(self.values))
        self.assertEqual(self.features_obj.max, np.max(self.values))
        self.assertEqual(self.features_obj.sum, np.sum(self.values))
        self.assertEqual(self.features_obj.first, self.values[0])
        self.assertEqual(self.features_obj.last, self.values[-1])
        
        # Test more complex statistical properties
        self.assertAlmostEqual(self.features_obj.std, np.std(self.values))
        self.assertAlmostEqual(self.features_obj.var, np.var(self.values))
        
    def test_batch_compress(self):
        # Test default batch compression
        compressed = self.features_obj.batch_compress().compress()
        expected_ops = ['mean', 'std', 'min', 'max', 'skew', 'kurtosis', 'variance', 'first', 'last', 'median', 'sum']
        
        # Verify each operation exists
        for op in expected_ops:
            self.assertIn(op, compressed.feature_names)
            
    def test_batch_compress_custom(self):
        # Test custom compression
        def custom_op(x): 
            return x.sum()
        custom_op.__name__ = "custom_sum"
        
        # Test with only custom compressor
        compressed = self.features_obj.batch_compress(
            common_operations=False,
            custom_compressors=[custom_op]
        ).compress()
        
        # Verify the custom operation was added and used
        self.assertIn("custom_sum", compressed.feature_names)
        
        # Test mixing custom and common operations
        compressed = self.features_obj.batch_compress(
            common_operations=True,
            custom_compressors=[custom_op]
        ).compress()
        
        # Verify all default operations plus custom one are present
        expected_ops = ['mean', 'std', 'min', 'max', 'skew', 'kurtosis', 'variance', 'first', 'last', 'median', 'sum', 'custom_sum']
        for op in expected_ops:
            self.assertIn(op, compressed.feature_names)
        
if __name__ == '__main__':
    unittest.main() 