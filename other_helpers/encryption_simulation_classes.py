# Pretend class for Right-Left Order Revealing Encryption. Only supports the allowed actions.
class ORE_Number:
    def __init__(self, number, enc_type):
        self.number = number
        if enc_type == 'L' or enc_type == 'R':
            self.enc_type = enc_type
        else:
            raise ValueError('Unknown Type "%s". Suppoted types are "L" and "R".' % enc_type)
        return
    
    def __eq__(self, other):
        if self.enc_type == other.enc_type:
            raise TypeError('Comparison not supported between encryptions of the same type. Both encryption are of type "%s".' % self.enc_type)
        return self.number == other.number
    
    def __lt__(self, other):
        if self.enc_type == other.enc_type:
            raise TypeError('Comparison not supported between encryptions of the same type. Both encryption are of type "%s".' % self.enc_type)
        return self.number < other.number
    
    def __gt__(self, other):
        if self.enc_type == other.enc_type:
            raise TypeError('Comparison not supported between encryptions of the same type. Both encryption are of type "%s".' % self.enc_type)
        return self.number > other.number


# Pretend class for Additive partially homomorphic encripted numbers. Ignores real->int encoding, and only supports allowed operations.
class Add_PHE_Number:
    def __init__(self, number, mults=0):
        self.number = number
        self.mults = mults
    
    def __add__(self, other):
        res = None
        if isinstance(other, Add_PHE_Number):
            res = Add_PHE_Number(self.number + other.number)
        else:
            res = Add_PHE_Number(self.number + other)
        return res
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        if isinstance(other, Add_PHE_Number):
            raise('Cannot multiply additive homomorphic numbers!')
        return Add_PHE_Number(self.number*other, mults=self.mults+1)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def get_number(self):
        return self.number