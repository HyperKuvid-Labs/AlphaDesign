import { useNavigate } from 'react-router-dom';

const Progress = () => {
  const navigate = useNavigate();
  const milestones = [
    {
      id: 1,
      title: "Reinforcement learning implementation",
      description: "A hybrid AI system that combines genetic algorithms and reinforcement learning to optimize Formula 1 front wing aerodynamics. The system explores wing geometries through evolutionary algorithms, then refines designs using RL agents trained on CFD simulations, all while maintaining strict FIA regulatory compliance.",
      image: "/milestone-1.jpg",
      status: "completed",
      date: "Week 1-2"
    },
    {
      id: 2,
      title: "Smoothening and bezier curve implementation",
      description: "The project introduces Bezier curve parameterization for smooth F1 wing profiles, intelligent CFD scheduling to cut simulation costs, and physics-informed rewards for RL agents. These innovations enable the system to achieve high aerodynamic fitness scores and regulatory compliance.",
      image: "/milestone-2.jpg", 
      status: "completed",
      date: "Week 3-4"
    },
    {
      id: 3,
      title: "DDPG-Based RL for Continuous Aerodynamic Optimization",
      description: "This milestone introduces the use of Deep Deterministic Policy Gradient (DDPG) to enable reinforcement learning in continuous action spaces for aerodynamic optimization, moving beyond traditional RL algorithms that are limited to discrete actions.",
      image: "/milestone-3.jpg",
      status: "in-progress",
      date: "Week 5-6"
    },
    {
      id: 4,
      title: "Modular F1 Car Generation Framework",
      description: "This milestone extends the automated design process from the front wing to the entire F1 car, enabling modular generation of all car components with high accuracy and quality.",
      image: "/milestone-4.jpg",
      status: "upcoming",
      date: "Week 7-8"
    }
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-400 border-green-400';
      case 'in-progress': return 'text-secondary border-secondary';
      case 'upcoming': return 'text-muted-foreground border-muted';
      default: return 'text-muted-foreground border-muted';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return '✓';
      case 'in-progress': return '⚡';
      case 'upcoming': return '○';
      default: return '○';
    }
  };

  return (
    <div className="min-h-screen bg-background py-20">
      <div className="container mx-auto px-4">
        {/* Header */}
        <div className="text-center mb-16">
          <h1 className="text-4xl md:text-6xl font-rajdhani font-bold text-foreground mb-4">
            Project Progress
          </h1>
          <p className="text-xl font-inter text-muted-foreground max-w-2xl mx-auto">
            Track our journey from concept to completion through detailed milestones 
            and development phases.
          </p>
        </div>

        {/* Progress Timeline */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 max-w-6xl mx-auto">
          {milestones.map((milestone, index) => (
            <div 
              key={milestone.id}
              className="timeline-card group cursor-pointer"
              style={{ animationDelay: `${index * 0.1}s` }}
              onClick={() => navigate(`/progress/${milestone.id}`)}
            >
              {/* Status Badge */}
              {/* <div className={`inline-flex items-center space-x-2 px-3 py-1 rounded-full text-sm font-medium border mb-4 ${getStatusColor(milestone.status)}`}>
                <span>{getStatusIcon(milestone.status)}</span>
                <span className="capitalize font-rajdhani">{milestone.status.replace('-', ' ')}</span>
              </div> */}

              {/* Milestone Image */}
              <div className="relative overflow-hidden rounded-lg mb-6 aspect-video">
                <img 
                  src={milestone.image} 
                  alt={milestone.title}
                  className="w-full h-full object-cover transform group-hover:scale-105 transition-racing"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent" />
              </div>

              {/* Content */}
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-2xl font-rajdhani font-semibold text-foreground">
                    {milestone.title}  
                  </h3>
                  {/* <span className="text-sm font-inter text-primary font-medium">
                    {milestone.date}
                  </span> */}
                </div>
                
                <p className="text-muted-foreground font-inter leading-relaxed">
                  {milestone.description}
                </p>
              </div>

              {/* Progress Indicator */}
              {/* <div className="mt-6 pt-4 border-t border-border">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-rajdhani font-medium text-foreground">
                    Milestone {milestone.id} of 4
                  </span>
                  <div className="flex space-x-1">
                    {[1, 2, 3, 4].map((step) => (
                      <div 
                        key={step}
                        className={`w-2 h-2 rounded-full transition-colors ${
                          step <= milestone.id 
                            ? 'bg-primary' 
                            : 'bg-muted'
                        }`}
                      />
                    ))}
                  </div>
                </div>
              </div> */}
            </div>
          ))}
        </div>

      </div>
    </div>
  );
};

export default Progress;