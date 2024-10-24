from pathlib import Path
import tempfile
from llm_foundation import logger
from llm_foundation.agent_types import Persona, Role

def test_persona_to_yaml():
    role1 = Role(name="role1", description="Role 1 Description", agent_system_message="Agent System Message")
    role2 = Role(name="role2", description="Role 2 Description", agent_system_message="Agent System Message again")
    roles_dict= {"role1": role1, "role2": role2}
    persona = Persona(name="Test Persona", roles=roles_dict)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        
        filename="test_persona.yaml"

        persona.to_yaml_file(Path(temp_dir), filename)
        
        same_persona = Persona.from_yaml_file(str(Path(temp_dir, filename)))

        # Assertions
        assert persona.name == same_persona.name, "Not the same name"
        assert len(persona.roles) == len(same_persona.roles), "Not the same number of roles"
        
        for role_name, role in persona.roles.items():
            same_role = same_persona.roles.get(role_name)
            assert same_role is not None, f"Role {role_name} not found in deserialized persona"
            assert role.name == same_role.name, f"Role name mismatch for {role_name}"
            assert role.description == same_role.description, f"Role description mismatch for {role_name}"
            assert role.agent_system_message == same_role.agent_system_message, f"Agent system message mismatch for {role_name}"

def test_persona_from_yaml():
    yaml_content = """
    name: Test Persona
    roles:
        role1:
            name: role1
            description: Role 1 Description
            agent_system_message: Agent System Message
            tasks:
                - name: task1
                  description: This is task 1
                  expected_output: This is expected output for task 1
                - name: task2
                  description: This is task 2
                  expected_output: This is expected output for task 2
        role2:
            name: role2
            description: Role 2 Description
            agent_system_message: Agent System Message again
    """
    
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "test_persona.yaml"
        with open(file_path, 'w') as file:
            file.write(yaml_content)
        
        persona = Persona.from_yaml_file(str(file_path))

        # Assertions
        assert persona.name == "Test Persona", "Persona name mismatch"
        assert len(persona.roles) == 2, "Number of roles mismatch"
        
        role1 = persona.roles.get("role1")
        assert role1 is not None, "Role role1 not found"
        assert role1.name == "role1", "Role1 name mismatch"
        assert role1.description == "Role 1 Description", "Role1 description mismatch"
        assert role1.agent_system_message == "Agent System Message", "Role1 agent system message mismatch"
        assert len(role1.tasks) == 2, "Number of tasks in role1 mismatch"
        
        task1 = role1.tasks[0]
        assert task1.name == "task1", "Task1 name mismatch"
        assert task1.description == "This is task 1", "Task1 description mismatch"
        assert task1.expected_output == "This is expected output for task 1", "Task1 expected output mismatch"
        
        task2 = role1.tasks[1]
        assert task2.name == "task2", "Task2 name mismatch"
        assert task2.description == "This is task 2", "Task2 description mismatch"
        assert task2.expected_output == "This is expected output for task 2", "Task2 expected output mismatch"
                
        role2 = persona.roles.get("role2")
        assert role2 is not None, "Role role2 not found"
        assert role2.name == "role2", "Role2 name mismatch"
        assert role2.description == "Role 2 Description", "Role2 description mismatch"
        assert role2.agent_system_message == "Agent System Message again", "Role2 agent system message mismatch"
        assert len(role2.tasks) == 0, "Number of tasks in role2 mismatch"
