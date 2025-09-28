import alphaLogo from '@/assets/alphadesign-logo.png';
import f1CarBg from '@/assets/f1-car-bg.jpg';
import { useNavigate } from 'react-router-dom';

const Home = () => {
  const navigate = useNavigate();
  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* F1 Car Background */}
      <div className="absolute inset-0 w-full h-full">
        <img 
          src={f1CarBg} 
          alt="F1 Car Background" 
          className="w-full h-full object-cover opacity-10 select-none"
        />
        <div className="absolute inset-0 bg-gradient-to-br from-background/80 via-background/60 to-background/80" />
      </div>

      {/* Hero Content */}
      <div className="relative z-10 container mx-auto px-4 py-20">
        <div className="flex flex-col items-center justify-center min-h-[70vh] text-center space-y-8">
          
          {/* Project Name */}
          <div className="flex items-center justify-center space-x-6">
            {/* <img 
              src="/logo.png" 
              alt="Alpha Design Logo" 
              className="w-16 h-16 md:w-24 md:h-24 lg:w-32 lg:h-32"
            /> */}
            <h1 className="text-5xl md:text-7xl lg:text-8xl font-rajdhani font-bold text-foreground leading-tight">
              ALPHA DESIGN
            </h1>
          </div>

          {/* Project Description */}
          <p className="text-xl md:text-2xl font-inter max-w-3xl leading-relaxed text-muted-foreground">
            Revolutionary automotive design project inspired by the precision and performance of 
            Formula 1 engineering. Pushing the boundaries of aerodynamics, innovation, and 
            cutting-edge technology.
          </p>

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 mt-12">
            <button 
              className="btn-racing"
              onClick={() => navigate('/progress')}
            >
              Explore Progress
            </button>
            <button 
              className="btn-racing-outline"
              onClick={() => navigate('/model')}
            >
              View 3D Model
            </button>
          </div>

          {/* Scroll Indicator */}
          <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 animate-bounce">
            <div className="w-6 h-10 border-2 border-primary rounded-full flex justify-center">
              <div className="w-1 h-3 bg-primary rounded-full mt-2 animate-pulse"></div>
            </div>
          </div>
        </div>
      </div>

      {/* Stats Section */}
      <div className="relative z-10 bg-background py-20">
        <div className="container mx-auto px-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="text-center space-y-2">
              <div className="text-4xl md:text-5xl font-rajdhani font-bold text-foreground">
                9+
              </div>
              <div className="text-lg font-inter text-muted-foreground">
                Hours of Training
              </div>
            </div>
            
            <div className="text-center space-y-2">
              <div className="text-4xl md:text-5xl font-rajdhani font-bold text-foreground">
                4060
              </div>
              <div className="text-lg font-inter text-muted-foreground">
                RTX - GPU
              </div>
            </div>
            
            <div className="text-center space-y-2">
              <div className="text-4xl md:text-5xl font-rajdhani font-bold text-foreground">
                1950
              </div>
              <div className="text-lg font-inter text-muted-foreground">
                Achieved downforce (N)
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;