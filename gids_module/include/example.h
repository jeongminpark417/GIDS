
class Example
{
public:
  Example( double a )
    : _a( a)
  {
  }

  Example& operator+=( const Example& other )
  {
    _a += other._a;
    return *this;
  }

  void print();
private:
  double _a;
};


